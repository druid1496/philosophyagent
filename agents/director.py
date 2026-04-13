import json
import re
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


_ROUTING_PROMPT = """\
You are the **Debate Director** for a philosophical debate. You control who speaks next, optional provocation, \
and when to stop.

TOPIC: "{topic}"
PARTICIPANTS (exact names, use these strings verbatim): {philosophers}

RECENT SPEAKING ORDER (philosopher speeches only, oldest → newest; may repeat names):
{recent_philosopher_order}

LAST SPEAKER: {last_speaker}
THEIR LATEST STATEMENT (excerpt):
\"\"\"
{last_statement_excerpt}
\"\"\"

DEBATE FRICTION LOG (recent lines):
{debate_history}

RECENT TRANSCRIPT (most recent last):
\"\"\"
{recent_exchange}
\"\"\"

PHILOSOPHER SPEECHES COMPLETED SO FAR (includes opening stand): {directed_cycles} / {max_turns} before the system forces a stop.

Routing policy (critical):
- Choose **next_speaker** for **substance**, not fairness. Pick whoever should answer the last claim: best critic, \
clearest opponent, or the one whose framework was left unaddressed.
- **Do not** route by rotating through PARTICIPANTS in list order, and do not default to “everyone gets a turn.” \
If the same pairing is still productive, keep pressing it; if a voice has gone quiet too long, pull them back in.
- If RECENT SPEAKING ORDER shows one person dominating, prefer someone else **unless** the text clearly needs that \
person again.
- In **bridge**, briefly say *why this thinker* in one clause (e.g. tension with last claim, ignored thesis).

Your duties (one JSON object only, no markdown fences):
1. **conclude_debate** (boolean): true if this clash should end now—repetition, no new friction, natural closure, \
or the cap above is already reached / would add nothing. Base on dialogue shape and pacing only.
2. **next_speaker** (string or null): if conclude_debate is false, the ONE participant who must speak next—must be \
different from LAST SPEAKER and must be an exact name from PARTICIPANTS. If conclude_debate is true, use null.
3. **bridge** (string): one or two sentences read aloud to the room—why this speaker goes next, OR why the debate stops.
4. **chaos_question** (string or null): if the last exchange was too polite or bloodless, a single sharp provocative \
question (1–2 sentences) the next speaker must engage; otherwise null. If conclude_debate is true, use null.

Respond with ONLY valid JSON in exactly this shape:
{{"conclude_debate": <bool>, "next_speaker": <string or null>, "bridge": "<string>", "chaos_question": <string or null>}}
"""


@dataclass
class RoutingDecision:
    conclude_debate: bool
    next_speaker: str | None
    bridge: str
    chaos_question: str | None


def _fallback_next(
    philosopher_names: list[str],
    last_speaker: str,
    directed_cycles: int,
) -> str | None:
    """When JSON is bad or next_speaker is invalid, avoid always picking the first name in list order."""
    others = [n for n in philosopher_names if n != last_speaker]
    if not others:
        return None
    return others[directed_cycles % len(others)]


def _parse_routing_json(
    text: str,
    philosopher_names: list[str],
    last_speaker: str,
    directed_cycles: int,
) -> RoutingDecision:
    raw = text.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        raw = m.group(0)
    data = json.loads(raw)
    raw_cd = data.get("conclude_debate")
    if isinstance(raw_cd, str):
        conclude = raw_cd.strip().lower() in ("true", "1", "yes")
    else:
        conclude = bool(raw_cd)
    bridge = str(data.get("bridge", "")).strip() or "(Routing.)"
    nxt = data.get("next_speaker")
    if nxt is not None and not isinstance(nxt, str):
        nxt = str(nxt).strip() if nxt else None
    if isinstance(nxt, str):
        nxt = nxt.strip() or None
    chaos = data.get("chaos_question")
    if chaos is not None and not isinstance(chaos, str):
        chaos = str(chaos).strip() if chaos else None
    if isinstance(chaos, str) and (not chaos or chaos.upper() == "NONE"):
        chaos = None

    fb = _fallback_next(philosopher_names, last_speaker, directed_cycles)
    if not conclude and nxt is not None:
        if nxt not in philosopher_names or nxt == last_speaker:
            nxt = fb
    if not conclude and nxt is None and fb is not None:
        nxt = fb

    return RoutingDecision(
        conclude_debate=conclude,
        next_speaker=None if conclude else nxt,
        bridge=bridge,
        chaos_question=chaos,
    )


class DirectorAgent:
    """Debate Director: who speaks next, optional chaos provocation, when to end."""

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm
        prompt = ChatPromptTemplate.from_messages([("human", _ROUTING_PROMPT)])
        self._chain = prompt | llm

    def routing_decision(
        self,
        topic: str,
        philosopher_names: list[str],
        last_speaker: str,
        last_statement: str,
        recent_exchange: str,
        debate_history: list[str],
        recent_philosopher_order: list[str],
        directed_cycles_completed: int,
        max_turns: int,
    ) -> RoutingDecision:
        excerpt = last_statement[:6000] if last_statement else "(empty)"
        hist = "\n".join(debate_history[-12:]) if debate_history else "(none yet)"
        order_txt = (
            " → ".join(recent_philosopher_order)
            if recent_philosopher_order
            else "(none yet)"
        )
        text = self._chain.invoke({
            "topic": topic,
            "philosophers": ", ".join(philosopher_names),
            "recent_philosopher_order": order_txt,
            "last_speaker": last_speaker,
            "last_statement_excerpt": excerpt,
            "debate_history": hist,
            "recent_exchange": recent_exchange.strip() or "(none)",
            "directed_cycles": directed_cycles_completed,
            "max_turns": max_turns,
        }).content
        try:
            return _parse_routing_json(
                text, philosopher_names, last_speaker, directed_cycles_completed,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            nxt = _fallback_next(
                philosopher_names, last_speaker, directed_cycles_completed,
            )
            return RoutingDecision(
                conclude_debate=False,
                next_speaker=nxt,
                bridge="Director could not parse routing; defaulting to the next available voice.",
                chaos_question=None,
            )

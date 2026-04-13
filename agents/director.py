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

DIRECTED CLASHES SO FAR (after opening): {directed_cycles} / {max_turns} allowed before the system forces a stop.

Your duties (one JSON object only, no markdown fences):
1. **conclude_debate** (boolean): true if this clash should end now—e.g. repetition, exhaustion of useful antagonists, \
natural closure, or the cap above is already reached / would add nothing. Base this on dialogue shape and pacing only.
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


def _parse_routing_json(
    text: str,
    philosopher_names: list[str],
    last_speaker: str,
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

    others = [n for n in philosopher_names if n != last_speaker]
    if not conclude and nxt is not None:
        if nxt not in philosopher_names or nxt == last_speaker:
            nxt = others[0] if others else None
    if not conclude and nxt is None and others:
        nxt = others[0]

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
        directed_cycles_completed: int,
        max_turns: int,
    ) -> RoutingDecision:
        excerpt = last_statement[:6000] if last_statement else "(empty)"
        hist = "\n".join(debate_history[-12:]) if debate_history else "(none yet)"
        text = self._chain.invoke({
            "topic": topic,
            "philosophers": ", ".join(philosopher_names),
            "last_speaker": last_speaker,
            "last_statement_excerpt": excerpt,
            "debate_history": hist,
            "recent_exchange": recent_exchange.strip() or "(none)",
            "directed_cycles": directed_cycles_completed,
            "max_turns": max_turns,
        }).content
        try:
            return _parse_routing_json(text, philosopher_names, last_speaker)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            others = [n for n in philosopher_names if n != last_speaker]
            nxt = others[0] if others else None
            return RoutingDecision(
                conclude_debate=False,
                next_speaker=nxt,
                bridge="Director could not parse routing; defaulting to the next available voice.",
                chaos_question=None,
            )

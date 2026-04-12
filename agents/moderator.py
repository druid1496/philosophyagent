from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


_OPENING_HUB_PROMPT = """\
You are the moderator opening a live philosophical debate on a contemporary topic.

TOPIC: "{topic}"
PARTICIPANTS: {philosophers}

INITIAL PROPONENT (must take the first stand): {initial_proponent}

In 130–170 words:
1. Name the topic sharply as a modern moral or political question.
2. Explain that a **Debate Director** (not you) will read each speech and decide who answers next—or when the \
melee should end—based on friction and pacing, not on scoring who is "right."
3. Address {initial_proponent} by name and demand a clear opening stand: thesis, one key reason, and what would count as refuting them.
4. {intermission_policy}
Do NOT invite everyone to speak in turn.
"""

_INTERMISSION_PROMPT = """\
You are the moderator at a CHECKPOINT in a philosophical debate.

TOPIC: "{topic}"
PARTICIPANTS: {philosophers}

This checkpoint fires after **{speech_count}** philosopher speeches (you appear every **{interval}** speeches).

DEBATE SO FAR (moderator, philosophers, director; most recent last):
\"\"\"
{transcript}
\"\"\"

In 130–170 words:
1. Synthesize the main positions and clashes since the last checkpoint (or since opening if this is the first).
2. Name the sharpest unresolved tension.
3. Offer one refocusing question for the room. Do **not** assign who speaks next—the Debate Director routes speakers.
"""

_CONCLUSION_PROMPT = """\
You are moderating a philosophical debate on: "{topic}"

FULL DEBATE TRANSCRIPT:
{transcript}

Write a conclusion (200–250 words):
1. Each philosopher's core position in 1–2 sentences.
2. The most fundamental point of disagreement.
3. Any surprising convergences.
4. What this debate illuminates about the topic that a single thinker could not.
"""


def _intermission_policy_text(interval: int) -> str:
    if interval <= 0:
        return (
            "Apart from this opening and the final synthesis, you will **not** step in mid-debate; "
            "the Debate Director alone routes speakers until the close."
        )
    return (
        f"Apart from opening and the final synthesis, you will step in every **{interval}** philosopher speeches "
        f"for a brief checkpoint—synthesis and a refocusing question—then the Debate Director resumes."
    )


class ModeratorAgent:
    def __init__(self, llm: ChatOpenAI):
        self._llm = llm

    def opening_hub(
        self,
        topic: str,
        philosopher_names: list[str],
        initial_proponent: str,
        moderator_intermission_every: int,
    ) -> str:
        chain = ChatPromptTemplate.from_messages([("human", _OPENING_HUB_PROMPT)]) | self._llm
        return chain.invoke({
            "topic": topic,
            "philosophers": ", ".join(philosopher_names),
            "initial_proponent": initial_proponent,
            "intermission_policy": _intermission_policy_text(moderator_intermission_every),
        }).content

    def intermission_checkpoint(
        self,
        topic: str,
        philosopher_names: list[str],
        transcript: str,
        speech_count: int,
        interval: int,
    ) -> str:
        chain = ChatPromptTemplate.from_messages([("human", _INTERMISSION_PROMPT)]) | self._llm
        return chain.invoke({
            "topic": topic,
            "philosophers": ", ".join(philosopher_names),
            "transcript": transcript,
            "speech_count": speech_count,
            "interval": interval,
        }).content

    def conclude(self, topic: str, all_messages: list[dict]) -> str:
        parts = []
        for msg in all_messages:
            if msg["speaker_type"] == "moderator":
                parts.append(f"[Moderator]: {msg['content']}")
            elif msg["speaker_type"] == "philosopher":
                parts.append(f"{msg['speaker_name']} (turn {msg['turn'] + 1}): {msg['content']}")
            elif msg["speaker_type"] == "director":
                parts.append(f"[Debate Director]: {msg['content']}")
        transcript = "\n\n".join(parts)

        chain = ChatPromptTemplate.from_messages([("human", _CONCLUSION_PROMPT)]) | self._llm
        return chain.invoke({
            "topic": topic,
            "transcript": transcript,
        }).content

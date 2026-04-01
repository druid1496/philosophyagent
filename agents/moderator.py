from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


_INTRO_PROMPT = """\
You are a moderator facilitating a rigorous philosophical debate.

TOPIC: "{topic}"
PARTICIPATING PHILOSOPHERS: {philosophers}

Introduce the debate. In 120–150 words:
1. Frame the philosophical significance of the topic.
2. Note briefly why each philosopher's perspective will be distinctive and valuable.
3. Pose one precise opening question to guide the first round.
"""

_TURN_SUMMARY_PROMPT = """\
You are moderating a philosophical debate on: "{topic}"

ROUND {turn_number} — STATEMENTS AND EVALUATIONS:
{turn_content}

In 100–120 words:
1. Summarize the key positions and the most interesting tension or disagreement that emerged.
2. Pose one focused follow-up question to sharpen the next round.
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


class ModeratorAgent:
    def __init__(self, llm: ChatOpenAI):
        self._llm = llm

    def introduce(self, topic: str, philosopher_names: list[str]) -> str:
        chain = ChatPromptTemplate.from_messages([("human", _INTRO_PROMPT)]) | self._llm
        return chain.invoke({
            "topic": topic,
            "philosophers": ", ".join(philosopher_names),
        }).content

    def summarize_turn(
        self, topic: str, turn_number: int, turn_messages: list[dict]
    ) -> str:
        lines = []
        for msg in turn_messages:
            if msg["speaker_type"] == "philosopher":
                lines.append(f"\n{msg['speaker_name']}:\n{msg['content']}")
            elif msg["speaker_type"] == "evaluator":
                lines.append(f"[Evaluation of {msg['speaker_name']}]:\n{msg['content']}")
        turn_content = "\n".join(lines)

        chain = ChatPromptTemplate.from_messages([("human", _TURN_SUMMARY_PROMPT)]) | self._llm
        return chain.invoke({
            "topic": topic,
            "turn_number": turn_number,
            "turn_content": turn_content,
        }).content

    def conclude(self, topic: str, all_messages: list[dict]) -> str:
        parts = []
        for msg in all_messages:
            if msg["speaker_type"] == "moderator":
                parts.append(f"[Moderator]: {msg['content']}")
            elif msg["speaker_type"] == "philosopher":
                parts.append(f"{msg['speaker_name']} (Round {msg['turn'] + 1}): {msg['content']}")
            elif msg["speaker_type"] == "evaluator":
                parts.append(f"  → Eval of {msg['speaker_name']}: {msg['content']}")
        transcript = "\n\n".join(parts)

        chain = ChatPromptTemplate.from_messages([("human", _CONCLUSION_PROMPT)]) | self._llm
        return chain.invoke({
            "topic": topic,
            "transcript": transcript,
        }).content

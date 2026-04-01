from typing import TypedDict, Annotated
import operator


class DebateMessage(TypedDict):
    turn: int
    speaker_type: str   # "moderator" | "philosopher" | "evaluator"
    speaker_name: str   # e.g. "Plato", "Moderator", "Evaluator"
    content: str


class DebateState(TypedDict):
    # Debate setup (populated at invocation, read-only during debate)
    topic: str
    max_turns: int
    philosopher_names: list[str]

    # Loop control (updated by nodes)
    current_turn: int           # 0-indexed round number
    current_philosopher_idx: int

    # Append-only transcript
    messages: Annotated[list[DebateMessage], operator.add]

    # Latest evaluation per philosopher (overwritten each turn)
    evaluations: dict[str, str]

    # Routing signal set by nodes, read by conditional edges
    phase: str  # "philosopher_speak" | "turn_summary" | "philosopher_speak" | "conclude"

    # Terminal flag
    debate_complete: bool

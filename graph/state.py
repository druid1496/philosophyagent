from typing import TypedDict, Annotated
import operator


class DebateMessage(TypedDict):
    turn: int
    speaker_type: str   # "moderator" | "philosopher" | "director"
    speaker_name: str   # e.g. "Plato", "Moderator", "Debate Director"
    content: str


class DebateState(TypedDict):
    """Hub debate: moderator opens/closes; Debate Director routes; philosophers clash."""

    topic: str
    max_turns: int
    # Insert a moderator checkpoint after every N philosopher speeches (0 = disabled).
    moderator_intermission_every: int
    philosopher_names: list[str]
    initial_proponent: str

    # Next speaker + rebuttal target (set by Debate Director before each non-opening speech)
    active_philosopher: str
    target_philosopher: str
    rebuttal_target_excerpt: str

    last_philosopher_speaker: str
    last_philosopher_content: str

    speech_sequence: int
    directed_cycles_completed: int

    messages: Annotated[list[DebateMessage], operator.add]

    debate_history: Annotated[list[str], operator.add]

    chaos_factor: str
    should_end_debate: bool

    debate_complete: bool

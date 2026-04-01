from langgraph.graph import StateGraph, START, END

from graph.state import DebateState, DebateMessage
from agents.philosopher import PhilosopherAgent
from agents.evaluator import EvaluatorAgent
from agents.moderator import ModeratorAgent


def _format_history(messages: list[DebateMessage]) -> str:
    if not messages:
        return "No prior exchanges."
    parts = []
    for msg in messages:
        turn_label = f"Round {msg['turn'] + 1}" if msg["speaker_type"] != "moderator" else "Moderator"
        parts.append(f"[{turn_label}] {msg['speaker_name']}: {msg['content']}")
    return "\n\n".join(parts)


def _print_divider(label: str = "") -> None:
    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label}")
        print("=" * 60)


def build_debate_graph(
    philosopher_agents: dict[str, PhilosopherAgent],
    evaluator: EvaluatorAgent,
    moderator: ModeratorAgent,
):
    # ------------------------------------------------------------------ nodes

    def node_moderator_introduce(state: DebateState) -> dict:
        intro = moderator.introduce(state["topic"], state["philosopher_names"])
        _print_divider("MODERATOR — Introduction")
        print(intro)
        return {
            "messages": [DebateMessage(
                turn=0,
                speaker_type="moderator",
                speaker_name="Moderator",
                content=intro,
            )],
            "phase": "philosopher_speak",
        }

    def node_philosopher_speak(state: DebateState) -> dict:
        idx = state["current_philosopher_idx"]
        name = state["philosopher_names"][idx]
        turn = state["current_turn"]

        history = _format_history(state["messages"])
        last_eval = state["evaluations"].get(name, "")

        response = philosopher_agents[name].respond(state["topic"], history, last_eval)
        _print_divider(f"{name}  —  Round {turn + 1}")
        print(response)

        return {
            "messages": [DebateMessage(
                turn=turn,
                speaker_type="philosopher",
                speaker_name=name,
                content=response,
            )],
        }

    def node_evaluator(state: DebateState) -> dict:
        idx = state["current_philosopher_idx"]
        name = state["philosopher_names"][idx]

        # Collect this philosopher's statements in chronological order
        phil_msgs = [
            m["content"]
            for m in state["messages"]
            if m["speaker_type"] == "philosopher" and m["speaker_name"] == name
        ]
        latest = phil_msgs[-1] if phil_msgs else ""
        previous = phil_msgs[:-1]

        evaluation = evaluator.evaluate(name, state["topic"], latest, previous)
        print(f"\n  [Evaluator → {name}]\n  {evaluation}")

        # Advance index; if all philosophers done, route to turn summary
        next_idx = idx + 1
        all_done = next_idx >= len(state["philosopher_names"])

        return {
            "messages": [DebateMessage(
                turn=state["current_turn"],
                speaker_type="evaluator",
                speaker_name=name,
                content=evaluation,
            )],
            "evaluations": {**state["evaluations"], name: evaluation},
            "current_philosopher_idx": 0 if all_done else next_idx,
            "phase": "turn_summary" if all_done else "philosopher_speak",
        }

    def node_moderator_turn_summary(state: DebateState) -> dict:
        turn = state["current_turn"]
        turn_msgs = [
            m for m in state["messages"]
            if m["turn"] == turn and m["speaker_type"] != "moderator"
        ]
        summary = moderator.summarize_turn(state["topic"], turn + 1, turn_msgs)
        _print_divider(f"MODERATOR — Round {turn + 1} Summary")
        print(summary)

        next_turn = turn + 1
        is_last = next_turn >= state["max_turns"]

        return {
            "messages": [DebateMessage(
                turn=turn,
                speaker_type="moderator",
                speaker_name="Moderator",
                content=summary,
            )],
            "current_turn": next_turn,
            "phase": "conclude" if is_last else "philosopher_speak",
        }

    def node_moderator_conclude(state: DebateState) -> dict:
        conclusion = moderator.conclude(state["topic"], state["messages"])
        _print_divider("MODERATOR — Conclusion")
        print(conclusion)
        _print_divider()

        return {
            "messages": [DebateMessage(
                turn=state["current_turn"],
                speaker_type="moderator",
                speaker_name="Moderator",
                content=conclusion,
            )],
            "debate_complete": True,
        }

    # --------------------------------------------------------------- routing

    def route_after_evaluator(state: DebateState) -> str:
        return state["phase"]  # "philosopher_speak" or "turn_summary"

    def route_after_turn_summary(state: DebateState) -> str:
        return state["phase"]  # "philosopher_speak" or "conclude"

    # ----------------------------------------------------------------- graph

    g = StateGraph(DebateState)

    g.add_node("moderator_introduce", node_moderator_introduce)
    g.add_node("philosopher_speak", node_philosopher_speak)
    g.add_node("evaluator", node_evaluator)
    g.add_node("moderator_turn_summary", node_moderator_turn_summary)
    g.add_node("moderator_conclude", node_moderator_conclude)

    g.add_edge(START, "moderator_introduce")
    g.add_edge("moderator_introduce", "philosopher_speak")
    g.add_edge("philosopher_speak", "evaluator")
    g.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "philosopher_speak": "philosopher_speak",
            "turn_summary": "moderator_turn_summary",
        },
    )
    g.add_conditional_edges(
        "moderator_turn_summary",
        route_after_turn_summary,
        {
            "philosopher_speak": "philosopher_speak",
            "conclude": "moderator_conclude",
        },
    )
    g.add_edge("moderator_conclude", END)

    return g.compile()

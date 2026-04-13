from langgraph.graph import StateGraph, START, END

from graph.state import DebateState, DebateMessage
from agents.philosopher import PhilosopherAgent
from agents.director import DirectorAgent
from agents.moderator import ModeratorAgent


def _format_history(messages: list[DebateMessage]) -> str:
    if not messages:
        return "No prior exchanges."
    parts = []
    for msg in messages:
        parts.append(f"[{msg['speaker_name']} ({msg['speaker_type']})]: {msg['content']}")
    return "\n\n".join(parts)


def _recent_exchange_text(messages: list[DebateMessage], limit: int = 10) -> str:
    tail = messages[-limit:]
    lines = []
    for m in tail:
        excerpt = m["content"][:1800] + ("…" if len(m["content"]) > 1800 else "")
        lines.append(f"{m['speaker_name']} ({m['speaker_type']}): {excerpt}")
    return "\n\n".join(lines)


def _recent_philosopher_order(messages: list[DebateMessage], limit: int = 12) -> list[str]:
    names: list[str] = []
    for m in messages:
        if m["speaker_type"] == "philosopher":
            names.append(m["speaker_name"])
    return names[-limit:]


def _print_divider(label: str = "") -> None:
    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label}")
        print("=" * 60)


def build_debate_graph(
    philosopher_agents: dict[str, PhilosopherAgent],
    director: DirectorAgent,
    moderator: ModeratorAgent,
):
    def node_moderator_opening(state: DebateState) -> dict:
        intro = moderator.opening_hub(
            state["topic"],
            state["philosopher_names"],
            state["initial_proponent"],
            state["moderator_intermission_every"],
        )
        _print_divider("MODERATOR — Opening")
        print(intro)
        return {
            "messages": [DebateMessage(
                turn=0,
                speaker_type="moderator",
                speaker_name="Moderator",
                content=intro,
            )],
        }

    def node_philosopher_speak(state: DebateState) -> dict:
        name = state["active_philosopher"]
        turn_idx = state["speech_sequence"]
        history = _format_history(state["messages"])

        response = philosopher_agents[name].respond(
            state["topic"],
            history,
            target_philosopher=state["target_philosopher"],
            opponent_excerpt=state["rebuttal_target_excerpt"],
            chaos_factor=state["chaos_factor"],
        )
        _print_divider(f"{name}  —  Debate turn {turn_idx + 1}")
        print(response)

        return {
            "messages": [DebateMessage(
                turn=turn_idx,
                speaker_type="philosopher",
                speaker_name=name,
                content=response,
            )],
            "last_philosopher_speaker": name,
            "last_philosopher_content": response,
            "speech_sequence": turn_idx + 1,
            "chaos_factor": "",
        }

    def node_moderator_intermission(state: DebateState) -> dict:
        iv = state["moderator_intermission_every"]
        seq = state["speech_sequence"]
        transcript = _format_history(state["messages"])
        if len(transcript) > 12000:
            transcript = transcript[-12000:]

        body = moderator.intermission_checkpoint(
            state["topic"],
            state["philosopher_names"],
            transcript,
            speech_count=seq,
            interval=iv,
        )
        header = f"— Moderator checkpoint (after {seq} philosopher speeches; every {iv}) —"
        content = f"{header}\n\n{body}"
        _print_divider(f"MODERATOR — Checkpoint (every {iv} speeches)")
        print(content)

        return {
            "messages": [DebateMessage(
                turn=seq,
                speaker_type="moderator",
                speaker_name="Moderator",
                content=content,
            )],
        }

    def node_debate_director(state: DebateState) -> dict:
        name = state["last_philosopher_speaker"]
        speech_seq = state["speech_sequence"]
        last_text = state["last_philosopher_content"]

        if len(state["philosopher_names"]) < 2:
            line = "Only one participant — routing disabled; ending debate."
            _print_divider("DEBATE DIRECTOR")
            print(line)
            return {
                "messages": [DebateMessage(
                    turn=max(0, speech_seq - 1),
                    speaker_type="director",
                    speaker_name="Debate Director",
                    content=line,
                )],
                "should_end_debate": True,
            }

        new_directed = state["directed_cycles_completed"] + 1

        if new_directed >= state["max_turns"]:
            line = (
                f"Philosopher speech cap reached ({state['max_turns']} total, including opening stand). "
                f"Closing debate.\n{state['topic'][:80]}…"
            )
            _print_divider("DEBATE DIRECTOR — Cap stop")
            print(line)
            return {
                "messages": [DebateMessage(
                    turn=max(0, speech_seq - 1),
                    speaker_type="director",
                    speaker_name="Debate Director",
                    content=line,
                )],
                "directed_cycles_completed": new_directed,
                "should_end_debate": True,
            }

        decision = director.routing_decision(
            state["topic"],
            state["philosopher_names"],
            name,
            last_text,
            _recent_exchange_text(state["messages"]),
            list(state["debate_history"]),
            _recent_philosopher_order(state["messages"]),
            new_directed,
            state["max_turns"],
        )

        log_lines = [decision.bridge]
        if decision.conclude_debate:
            log_lines.append("→ Ending debate (director signals conclude).")
        elif decision.next_speaker:
            log_lines.append(f"→ Next speaker: {decision.next_speaker} (vs {name}).")
        if decision.chaos_question:
            log_lines.append(f"→ Chaos provocation for next speech: {decision.chaos_question}")
        log_body = "\n".join(log_lines)

        _print_divider("DEBATE DIRECTOR")
        print(log_body)

        out: dict = {
            "messages": [DebateMessage(
                turn=max(0, speech_seq - 1),
                speaker_type="director",
                speaker_name="Debate Director",
                content=log_body,
            )],
            "directed_cycles_completed": new_directed,
            "should_end_debate": bool(decision.conclude_debate),
        }

        if not decision.conclude_debate and decision.next_speaker:
            excerpt = last_text[:4000] if last_text else ""
            out["active_philosopher"] = decision.next_speaker
            out["target_philosopher"] = name
            out["rebuttal_target_excerpt"] = excerpt
            chaos = (decision.chaos_question or "").strip()
            if chaos:
                out["chaos_factor"] = chaos
                out["debate_history"] = [
                    f"{decision.next_speaker} vs {name} (chaos: {chaos[:120]}{'…' if len(chaos) > 120 else ''})",
                ]
            else:
                out["debate_history"] = [f"{decision.next_speaker} vs {name}"]

        return out

    def node_moderator_conclude(state: DebateState) -> dict:
        conclusion = moderator.conclude(state["topic"], state["messages"])
        _print_divider("MODERATOR — Conclusion")
        print(conclusion)
        _print_divider()

        return {
            "messages": [DebateMessage(
                turn=state["speech_sequence"],
                speaker_type="moderator",
                speaker_name="Moderator",
                content=conclusion,
            )],
            "debate_complete": True,
        }

    def route_after_director(state: DebateState) -> str:
        if len(state["philosopher_names"]) < 2:
            return "conclude"
        if state["directed_cycles_completed"] >= state["max_turns"]:
            return "conclude"
        if state["should_end_debate"]:
            return "conclude"
        return "philosopher_speak"

    def route_after_philosopher(state: DebateState) -> str:
        every = state["moderator_intermission_every"]
        seq = state["speech_sequence"]
        if every > 0 and seq > 0 and seq % every == 0:
            return "intermission"
        return "director"

    g = StateGraph(DebateState)

    g.add_node("moderator_opening", node_moderator_opening)
    g.add_node("philosopher_speak", node_philosopher_speak)
    g.add_node("moderator_intermission", node_moderator_intermission)
    g.add_node("debate_director", node_debate_director)
    g.add_node("moderator_conclude", node_moderator_conclude)

    g.add_edge(START, "moderator_opening")
    g.add_edge("moderator_opening", "philosopher_speak")
    g.add_conditional_edges(
        "philosopher_speak",
        route_after_philosopher,
        {
            "intermission": "moderator_intermission",
            "director": "debate_director",
        },
    )
    g.add_edge("moderator_intermission", "debate_director")
    g.add_conditional_edges(
        "debate_director",
        route_after_director,
        {
            "philosopher_speak": "philosopher_speak",
            "conclude": "moderator_conclude",
        },
    )
    g.add_edge("moderator_conclude", END)

    return g.compile()

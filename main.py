"""
Philosophy Debate Agent
=======================
Run a multi-turn philosophical debate between RAG-backed philosopher agents,
a debate director, and a moderator, all orchestrated via LangGraph.

Usage:
    python main.py
    python main.py --topic "Is democracy the best form of government?" --turns 3
    python main.py --philosophers Plato Nietzsche --turns 2
    python main.py --moderator-every 4
"""

import argparse
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import (
    PHILOSOPHERS,
    DEBATE_MODEL,
    EMBEDDING_MODEL,
    AGENT_TEMPERATURE,
    DIRECTOR_TEMPERATURE,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODERATOR_INTERMISSION_EVERY,
    DEFAULT_TOPIC,
)
from rag.retriever import setup_philosopher_rag
from agents.philosopher import PhilosopherAgent
from agents.director import DirectorAgent
from agents.moderator import ModeratorAgent
from graph.debate_graph import build_debate_graph
from graph.state import DebateState


def parse_args():
    parser = argparse.ArgumentParser(description="Run a philosophical debate between AI agents.")
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help="The debate topic.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=(
            "Maximum philosopher speeches in the debate, including the opening stand "
            "(each speech is followed by the Debate Director unless the cap is reached). "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--moderator-every",
        type=int,
        default=DEFAULT_MODERATOR_INTERMISSION_EVERY,
        metavar="N",
        help=(
            "Insert a moderator checkpoint every N philosopher speeches (0 = off). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--philosophers",
        nargs="+",
        default=list(PHILOSOPHERS.keys()),
        choices=list(PHILOSOPHERS.keys()),
        help="Which philosophers to include (default: all).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also save the full transcript as a JSON file.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Ignore cached FAISS indexes and rebuild from source texts.",
    )
    parser.add_argument(
        "--export-graph",
        nargs="?",
        const="debate_graph.png",
        default=None,
        metavar="PNG_PATH",
        help=(
            "Export the compiled LangGraph diagram as a PNG. "
            "If path is omitted, uses debate_graph.png."
        ),
    )
    return parser.parse_args()


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Copy .env.example to .env and add your key."
        )

    args = parse_args()

    if args.moderator_every < 0:
        raise ValueError("--moderator-every must be >= 0 (use 0 to disable checkpoints).")

    # Validate philosopher selection
    for name in args.philosophers:
        if name not in PHILOSOPHERS:
            raise ValueError(f"Unknown philosopher: {name}. Available: {list(PHILOSOPHERS.keys())}")

    selected = {name: PHILOSOPHERS[name] for name in args.philosophers}
    names = list(selected.keys())

    print(f"\n{'#' * 60}")
    print("  PHILOSOPHICAL DEBATE")
    print(f"  Topic   : {args.topic}")
    print(f"  Max philosopher speeches (incl. opening): {args.turns}")
    print(f"  Opening proponent (first in list): {names[0]}")
    print(f"  Agents  : {', '.join(selected.keys())}")
    print(f"  Model   : {DEBATE_MODEL}")
    if args.moderator_every > 0:
        print(f"  Moderator checkpoint: every {args.moderator_every} philosopher speeches")
    else:
        print("  Moderator checkpoint: off")
    print(f"{'#' * 60}\n")

    # ----------------------------------------------------------------- Setup RAG
    print("Loading or building persistent vector stores (vector_stores/<philosopher>/)...")
    retrievers = {}
    for name, cfg in selected.items():
        action = "rebuilding" if args.rebuild_index else "loading / indexing"
        print(f"  [{name}] {action} {cfg['text_file']}")
        retrievers[name] = setup_philosopher_rag(
            name,
            cfg["text_file"],
            EMBEDDING_MODEL,
            force_rebuild=args.rebuild_index,
        )
    print("Done.\n")

    # ---------------------------------------------------------------- Agents
    llm = ChatOpenAI(model=DEBATE_MODEL, temperature=AGENT_TEMPERATURE)
    director_llm = ChatOpenAI(model=DEBATE_MODEL, temperature=DIRECTOR_TEMPERATURE)

    philosopher_agents = {
        name: PhilosopherAgent(
            name=name,
            description=cfg["description"],
            era=cfg["era"],
            retriever=retrievers[name],
            llm=llm,
        )
        for name, cfg in selected.items()
    }
    director = DirectorAgent(llm=director_llm)
    moderator = ModeratorAgent(llm=llm)

    # ----------------------------------------------------------------- Graph
    debate_graph = build_debate_graph(philosopher_agents, director, moderator)

    if args.export_graph:
        try:
            png_bytes = debate_graph.get_graph().draw_mermaid_png()
            with open(args.export_graph, "wb") as f:
                f.write(png_bytes)
            print(f"Graph PNG exported to {args.export_graph}")
        except Exception as e:
            print(f"Warning: could not export graph PNG ({e})")

    # ------------------------------------------------------------ Initial state (hub-and-spoke)
    proponent = names[0]
    initial_state = DebateState(
        topic=args.topic,
        max_turns=args.turns,
        moderator_intermission_every=args.moderator_every,
        philosopher_names=names,
        initial_proponent=proponent,
        active_philosopher=proponent,
        target_philosopher="",
        rebuttal_target_excerpt="",
        last_philosopher_speaker="",
        last_philosopher_content="",
        speech_sequence=0,
        directed_cycles_completed=0,
        messages=[],
        debate_history=[],
        chaos_factor="",
        should_end_debate=False,
        debate_complete=False,
    )

    # ------------------------------------------------------------------ Run
    final_state = debate_graph.invoke(initial_state)

    print(f"\n{'#' * 60}")
    print("  DEBATE COMPLETE")
    print(f"  Total messages: {len(final_state['messages'])}")
    print(f"{'#' * 60}\n")

    # ------------------------------------------------------------------ Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Always write a human-readable .txt transcript
    txt_filename = f"debate_{timestamp}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("PHILOSOPHICAL DEBATE TRANSCRIPT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Topic      : {final_state['topic']}\n")
        f.write(f"Philosophers: {', '.join(final_state['philosopher_names'])}\n")
        f.write(f"Max philosopher speeches (incl. opening): {args.turns}\n")
        f.write(
            f"Moderator checkpoint: every {args.moderator_every} philosopher speeches"
            f"{' (off)' if args.moderator_every == 0 else ''}\n"
        )
        f.write(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        current_round = -1
        for msg in final_state["messages"]:
            round_num = msg["turn"] + 1

            # Print round header when a new round starts (for non-moderator intro)
            if msg["speaker_type"] == "philosopher" and msg["turn"] != current_round:
                current_round = msg["turn"]
                f.write(f"\n{'─' * 60}\n")
                f.write(f"  ROUND {round_num}\n")
                f.write(f"{'─' * 60}\n\n")

            if msg["speaker_type"] == "moderator":
                f.write(f"[MODERATOR]\n{msg['content']}\n\n")
            elif msg["speaker_type"] == "philosopher":
                f.write(f"[{msg['speaker_name'].upper()}]\n{msg['content']}\n\n")
            elif msg["speaker_type"] == "director":
                f.write(f"[DEBATE DIRECTOR]\n{msg['content']}\n\n")

    print(f"Transcript saved to {txt_filename}")

    # Optionally also save JSON
    if args.save:
        json_filename = f"debate_{timestamp}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "topic": final_state["topic"],
                    "philosophers": final_state["philosopher_names"],
                    "initial_proponent": final_state["initial_proponent"],
                    "turns": args.turns,
                    "moderator_intermission_every": args.moderator_every,
                    "messages": final_state["messages"],
                    "debate_history": final_state["debate_history"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"JSON data saved to {json_filename}")

    return final_state


if __name__ == "__main__":
    main()

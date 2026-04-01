"""
Philosophy Debate Agent
=======================
Run a multi-turn philosophical debate between RAG-backed philosopher agents,
an evaluator, and a moderator, all orchestrated via LangGraph.

Usage:
    python main.py
    python main.py --topic "Is democracy the best form of government?" --turns 3
    python main.py --philosophers Plato Nietzsche --turns 2
"""

import argparse
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import PHILOSOPHERS, DEBATE_MODEL, EMBEDDING_MODEL, DEFAULT_MAX_TURNS, DEFAULT_TOPIC
from rag.retriever import setup_philosopher_rag
from agents.philosopher import PhilosopherAgent
from agents.evaluator import EvaluatorAgent
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
        help="Number of debate rounds (default: 2).",
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
    return parser.parse_args()


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Copy .env.example to .env and add your key."
        )

    args = parse_args()

    # Validate philosopher selection
    for name in args.philosophers:
        if name not in PHILOSOPHERS:
            raise ValueError(f"Unknown philosopher: {name}. Available: {list(PHILOSOPHERS.keys())}")

    selected = {name: PHILOSOPHERS[name] for name in args.philosophers}

    print(f"\n{'#' * 60}")
    print("  PHILOSOPHICAL DEBATE")
    print(f"  Topic   : {args.topic}")
    print(f"  Rounds  : {args.turns}")
    print(f"  Agents  : {', '.join(selected.keys())}")
    print(f"  Model   : {DEBATE_MODEL}")
    print(f"{'#' * 60}\n")

    # ----------------------------------------------------------------- Setup RAG
    print("Loading philosopher texts and building vector stores...")
    retrievers = {}
    for name, cfg in selected.items():
        print(f"  [{name}] indexing {cfg['text_file']}")
        retrievers[name] = setup_philosopher_rag(name, cfg["text_file"], EMBEDDING_MODEL)
    print("Done.\n")

    # ---------------------------------------------------------------- Agents
    llm = ChatOpenAI(model=DEBATE_MODEL, temperature=0.7)

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
    evaluator = EvaluatorAgent(llm=llm, retrievers=retrievers)
    moderator = ModeratorAgent(llm=llm)

    # ----------------------------------------------------------------- Graph
    debate_graph = build_debate_graph(philosopher_agents, evaluator, moderator)

    # ------------------------------------------------------------ Initial state
    initial_state = DebateState(
        topic=args.topic,
        max_turns=args.turns,
        philosopher_names=list(selected.keys()),
        current_turn=0,
        current_philosopher_idx=0,
        messages=[],
        evaluations={},
        phase="introduce",
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
        f.write(f"Rounds     : {args.turns}\n")
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
            elif msg["speaker_type"] == "evaluator":
                f.write(f"  ↳ Evaluation of {msg['speaker_name']}:\n")
                for line in msg["content"].splitlines():
                    f.write(f"  {line}\n")
                f.write("\n")

    print(f"Transcript saved to {txt_filename}")

    # Optionally also save JSON
    if args.save:
        json_filename = f"debate_{timestamp}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "topic": final_state["topic"],
                    "philosophers": final_state["philosopher_names"],
                    "turns": args.turns,
                    "messages": final_state["messages"],
                    "evaluations": final_state["evaluations"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"JSON data saved to {json_filename}")

    return final_state


if __name__ == "__main__":
    main()

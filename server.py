"""
Philosophy Debate Arena — FastAPI server
Run:  uvicorn server:app --reload --port 8000
or:   python server.py
"""

import asyncio
import json
import os
import queue
import threading
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Philosophy Debate Arena")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────── models ──

class DebateConfig(BaseModel):
    topic: str
    philosophers: list[str]
    turns: int = 6
    moderator_every: int = 4


# ──────────────────────────────────────────────────── debate logic ──

def _build_initial_state(config: DebateConfig, names: list[str]) -> dict:
    from graph.state import DebateState
    proponent = names[0]
    return DebateState(
        topic=config.topic,
        max_turns=config.turns,
        moderator_intermission_every=config.moderator_every,
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


def _run_debate_thread(config: DebateConfig, msg_queue: "queue.Queue[dict]") -> None:
    """Run the full debate in a worker thread, pushing events into msg_queue."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            msg_queue.put({"type": "error", "content": "OPENAI_API_KEY not set. Check your .env file."})
            msg_queue.put({"type": "done"})
            return

        from config import PHILOSOPHERS, DEBATE_MODEL, EMBEDDING_MODEL, AGENT_TEMPERATURE, DIRECTOR_TEMPERATURE
        from langchain_openai import ChatOpenAI
        from rag.retriever import setup_philosopher_rag
        from agents.philosopher import PhilosopherAgent
        from agents.director import DirectorAgent
        from agents.moderator import ModeratorAgent
        from graph.debate_graph import build_debate_graph

        valid = [p for p in config.philosophers if p in PHILOSOPHERS]
        if len(valid) < 2:
            msg_queue.put({"type": "error", "content": "Select at least 2 valid philosophers."})
            msg_queue.put({"type": "done"})
            return

        config.philosophers = valid
        selected = {name: PHILOSOPHERS[name] for name in valid}
        names = list(selected.keys())

        # ── Load vector stores ──
        msg_queue.put({"type": "status", "content": "Loading philosophical texts into memory…"})
        retrievers = {}
        for name, cfg in selected.items():
            msg_queue.put({"type": "status", "content": f"Summoning {name}'s writings…"})
            retrievers[name] = setup_philosopher_rag(name, cfg["text_file"], EMBEDDING_MODEL)

        # ── Build agents ──
        msg_queue.put({"type": "status", "content": "The philosophers take their seats…"})
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

        debate_graph = build_debate_graph(philosopher_agents, director, moderator)
        initial_state = _build_initial_state(config, names)

        msg_queue.put({"type": "debate_started", "data": {
            "topic": config.topic,
            "philosophers": names,
            "turns": config.turns,
            "moderator_every": config.moderator_every,
        }})

        # ── Stream nodes, collecting all messages for saving ──
        all_messages: list[dict] = []
        for event in debate_graph.stream(initial_state):
            for node_name, node_state in event.items():
                if node_state and "messages" in node_state:
                    for msg in node_state["messages"]:
                        d = dict(msg)
                        all_messages.append(d)
                        msg_queue.put({"type": "message", "data": d})

        # ── Save JSON transcript ──
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(f"debate_{timestamp}.json")
        save_path.write_text(
            json.dumps(
                {
                    "topic": config.topic,
                    "philosophers": names,
                    "turns": config.turns,
                    "moderator_intermission_every": config.moderator_every,
                    "messages": all_messages,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        msg_queue.put({"type": "done"})

    except Exception as exc:
        msg_queue.put({"type": "error", "content": f"{exc}\n\n{traceback.format_exc()}"})
        msg_queue.put({"type": "done"})


# ──────────────────────────────────────────────────────── routes ──

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found — add static/index.html</h1>", status_code=404)


@app.get("/philosophers")
async def get_philosophers():
    from config import PHILOSOPHERS
    return [
        {
            "name": name,
            "era": cfg["era"],
            "description": cfg["description"],
        }
        for name, cfg in PHILOSOPHERS.items()
    ]


@app.post("/debate/start")
async def start_debate(config: DebateConfig):
    msg_queue: queue.Queue = queue.Queue()

    thread = threading.Thread(
        target=_run_debate_thread,
        args=(config, msg_queue),
        daemon=True,
    )
    thread.start()

    async def generate():
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: msg_queue.get(timeout=300))
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") == "done":
                    break
            except queue.Empty:
                # Keep-alive heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/debates")
async def list_debates():
    debates = []
    for f in sorted(Path(".").glob("debate_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            msgs = [m for m in data.get("messages", []) if m.get("speaker_type") != "evaluator"]
            debates.append({
                "filename": f.name,
                "topic": data.get("topic", "Unknown topic"),
                "philosophers": data.get("philosophers", []),
                "message_count": len(msgs),
                "modified": f.stat().st_mtime,
            })
        except Exception:
            pass
    return debates


@app.get("/debates/{filename}")
async def get_debate(filename: str):
    if not filename.startswith("debate_") or not filename.endswith(".json"):
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    path = Path(filename)
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return json.loads(path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────── main ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

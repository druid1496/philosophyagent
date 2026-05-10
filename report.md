# Philosophy Debate Agent — Project Report


## Overview (TL;DR)

The Philosophy Debate Agent is a multi-agent system in which AI personas of historical philosophers (Plato, Aristotle, Kant, Mill, Nietzsche, Machiavelli) argue with each other about contemporary moral and political questions. Each philosopher is grounded in their own writings via Retrieval-Augmented Generation (RAG), so their claims must quote real source text rather than improvise from a stylistic caricature. A "Debate Director" agent watches the exchange and decides who answers next, when to inject a sharper provocation, and when to stop; a "Moderator" agent opens the debate, runs periodic synthesis checkpoints, and writes the final conclusion. The whole loop is orchestrated as a graph in LangGraph and exposed through both a CLI and a FastAPI web UI that streams the debate live to the browser.

---

## 1. Motivation — Why We Built It

We built this project to help people **think about today's hardest questions through the eyes of the great philosophers** — and, along the way, actually pick up the ideas those philosophers spent their lives developing.

The news is full of moral and political questions that don't have easy answers: Is AI a threat to human dignity? Is democracy still the best form of government? How much surveillance is too much? Most of us scroll past these without the time or background to think them through carefully. Meanwhile, the people who *did* spend their lives on questions like these — Plato, Aristotle, Kant, Mill, Nietzsche, Machiavelli — are stuck in textbooks that most people will never open.

We wanted to close that gap. The Philosophy Debate Agent lets a user pick a question they actually care about, choose two or more philosophers, and watch them argue it out in their own voice and with quotations from their own writings. The goal is twofold:

- **Make current events easier to think about.** Seeing Kant and Nietzsche disagree about AI is a much faster way to understand *what's at stake* in the AI debate than reading another op-ed. Each philosopher brings a different lens; the disagreement between the lenses is where the real questions live.
- **Make the philosophers' ideas stick.** It's one thing to read "Kant believed in the categorical imperative." It's another to watch Kant apply it to a problem you care about, in his own words, while Mill pushes back. The ideas land because you see them *do* something.

In short: this is a tool for **learning philosophy by watching it happen**, on the questions that matter right now.

---

## 2. Methods — How We Built It

### 2.1 System architecture at a glance

```
                ┌──────────────────────┐
                │      Moderator       │  opens, periodic checkpoints, conclusion
                └──────────┬───────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │         Philosopher Agents          │  one per thinker; each has its own
         │  Plato | Kant | Mill | Nietzsche…   │  FAISS index over its own text
         └──────────────────┬──────────────────┘
                            │  speech
                            ▼
                ┌──────────────────────┐
                │    Debate Director   │  picks next speaker, optional "chaos"
                └──────────┬───────────┘  provocation, decides when to stop
                           │
                           └─→ back to a philosopher  (loop until cap or stop)
```

The graph is hub-and-spoke: philosophers never call each other directly. Every philosopher speech goes back to the Director, who routes the next speaker. The Moderator is *not* in the routing loop — they only fire at structural moments (open, checkpoints, close).

### 2.2 The full debate flow

```
START
  └─> Moderator: opening (frames topic, names opening proponent)
        └─> Opening philosopher: first stand
              └─> [every N philosopher speeches → Moderator: checkpoint]
                    └─> Debate Director: who speaks next (or end)
                          └─> Next philosopher: rebuttal / advance
                                └─> ... loop ...
                                      └─> [until cap or director concludes]
                                            └─> Moderator: conclusion
                                                  └─> END
```

Two conditional edges control flow:

- After a philosopher speaks: if `speech_sequence % moderator_every == 0`, route to the moderator checkpoint; otherwise straight to the Director.
- After the Director speaks: if the cap is reached, or only one philosopher is in the room, or the Director signaled `conclude_debate=true`, route to the conclusion; otherwise loop back to a philosopher.

### 2.3 LangGraph state — what gets carried between nodes

We defined a single `DebateState` TypedDict that every node reads from and returns partial updates to. The non-obvious fields are:

| Field | Purpose |
|---|---|
| `active_philosopher` | Whose turn it is to speak next (set by Director). |
| `target_philosopher` | Who the next speaker must rebut/engage. |
| `rebuttal_target_excerpt` | The actual text excerpt the next speaker is responding to (avoids re-feeding the whole transcript). |
| `last_philosopher_speaker` / `last_philosopher_content` | What the Director just saw. Decoupled from `active_philosopher` because they refer to different turns. |
| `chaos_factor` | One-turn-only prompt injection from the Director. Cleared after each philosopher speech. |
| `messages` | The canonical, append-only transcript (uses `Annotated[..., operator.add]` so each node returns *only* its new messages). |
| `debate_history` | A compact routing log for the Director's "friction memory" (e.g. "Kant vs Mill (chaos: …)"). Separate from `messages` to keep the Director's prompt small. |
| `directed_cycles_completed` | Hard counter against `max_turns`; ends the debate even if the Director never decides to stop. |
| `should_end_debate` / `debate_complete` | Router booleans for the conditional edges. |

The split between `messages` (full transcript, used for context windows and final save) and `debate_history` (compact routing memory) is deliberate: the Director only needs to remember pairings and provocations, not full speeches, so feeding it the whole transcript would burn tokens for no signal.

### 2.4 Per-philosopher RAG — one vector store per voice

Each philosopher has their own corpus (`texts/plato.txt`, `texts/kant.txt`, etc.) and their own FAISS index built from it. The build pipeline:

1. **Chunking.** `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`, splitting on paragraph → line → space → character. The 200-char overlap matters for philosophical text where an argument crosses paragraph breaks.
2. **Embedding.** OpenAI `text-embedding-3-small`. Chosen for cost, not capability — the texts are small enough that retrieval quality is bottlenecked by chunk boundaries, not embedding model.
3. **Persistence with fingerprinting.** Each FAISS index is saved under `vector_stores/<Philosopher>/` along with a `build_meta.json` containing a SHA-256 fingerprint of `(text content + embedding model + chunk size + overlap)`. On startup, if the fingerprint matches, the index is loaded from disk; if any of those four things changed, it rebuilds. This avoids both stale indexes after editing a text file *and* paying the embedding cost on every cold start.
4. **Retrieval at debate time.** When a philosopher speaks, the query is built as `topic + "\n\nOpponent (X) claims, in part:\n<excerpt>"`. This is the key trick: retrieval is conditioned not just on the topic but on *what the opponent just said*. So when Kant is asked to rebut Nietzsche on AI dignity, the FAISS lookup pulls Kant passages relevant *to Nietzsche's specific framing*, not generic Kant-on-AI.

The retrieved passages are then injected into the philosopher's system prompt with a hard constraint: any verbatim quotation in the response must come *only* from those passages. This is the main defense against hallucinated citations.

### 2.5 The four agents

Each agent is a small class wrapping a `ChatPromptTemplate | ChatOpenAI` chain.

#### 2.5.1 PhilosopherAgent

System prompt locks the model into character (name, era, brief description of debate style and intellectual commitments). The user prompt provides:

- the topic,
- the named target thinker and an excerpt of their position (empty on the opening speech),
- the optional one-shot chaos provocation,
- the retrieved passages,
- the running debate history.

Hard rules in the prompt: stay in character, ground every claim in your real positions, include 1–2 verbatim quotations *only from the retrieved passages*, 150–250 words, no breaking character. The 150–250 word target keeps the rhythm of the debate from collapsing into monologues.

Temperature is `0.7` — high enough for stylistic distinctness across philosophers, low enough that they don't drift off-topic.

#### 2.5.2 DirectorAgent

The Director is the most prompt-engineered agent. It returns a single JSON object with four fields:

```json
{
  "conclude_debate": <bool>,
  "next_speaker":    <string or null>,
  "bridge":          "<short read-aloud rationale>",
  "chaos_question":  <string or null>
}
```

The routing policy is explicitly *anti-fairness*: the prompt tells the Director **not** to rotate through participants and **not** to default to "everyone gets a turn," but instead to pick whoever's framework was left unaddressed or whoever is the sharpest critic of the last claim. This was a direct response to early runs where the Director kept doing round-robin and the debate felt like a lineup, not an argument.

Robustness measures:

- **Lower temperature** (`0.45`) than the philosophers — routing decisions should be steadier, not creative.
- **JSON-extraction regex** on the response, so a stray markdown fence or prose preamble doesn't break parsing.
- **Validation:** if `next_speaker` isn't in the participant list, or equals the last speaker, fall back to a deterministic "next-other-philosopher" function based on `directed_cycles_completed`. This ensures the loop never wedges on a malformed model output.
- **Hard cap:** even if the Director never decides to stop, the graph forces a stop once `directed_cycles_completed >= max_turns`. The Director sees its progress against the cap in its prompt, so it knows to wrap up naturally as the budget runs out.

The "chaos provocation" is the fun part: when the Director judges the last exchange too polite or bloodless, it can attach a single sharp question that the next speaker is required to engage with. This is one-shot — the `chaos_factor` in state is cleared after the speech consumes it — so it acts as a nudge, not a permanent change of register.

#### 2.5.3 ModeratorAgent

Three different prompts for three different jobs:

- **Opening.** Frames the topic as a contemporary moral/political question, names the opening proponent, explains the role of the Debate Director, and (importantly) tells the room *not* to expect everyone to speak in turn.
- **Checkpoint.** Fires every N philosopher speeches (configurable, default 4, `0` to disable). Synthesizes the main positions and clashes since the last checkpoint and names the sharpest unresolved tension. Crucially, the prompt forbids the moderator from naming the next speaker — that's the Director's job.
- **Conclusion.** Consumes the full transcript and produces a 200–250 word synthesis: each philosopher's core position, the deepest disagreement, surprising convergences, and what the multi-voice debate illuminates that a single thinker would miss.

The transcript is trimmed to the last ~12,000 characters before being fed to the checkpoint to keep the context window manageable on long debates, but the conclusion uses the full transcript.

### 2.6 The web layer — streaming a graph over SSE

The CLI version writes everything to stdout, which is fine for development but doesn't make a watchable artifact. The web layer (`server.py` + `static/index.html`) wraps the same graph in a FastAPI app and streams it to the browser via Server-Sent Events.

The implementation pattern:

1. The browser POSTs `{topic, philosophers, turns, moderator_every}` to `/debate/start`.
2. The server spawns a worker thread that runs `debate_graph.stream(initial_state)` (LangGraph yields events for each node completion).
3. Each node's emitted messages are dropped into a `queue.Queue`.
4. The HTTP handler is an async generator that pulls from the queue and `yield`s `data: <json>\n\n` SSE frames. A 300-second `queue.get` timeout triggers a heartbeat frame so proxies don't drop the connection.
5. The browser consumes the stream and renders each message as it arrives — no polling, no websocket complexity.

A key design choice: the worker thread doesn't try to be clever about which messages to forward. It loops over the LangGraph stream events, and for any node that returned `messages`, it forwards each one. That means the SSE stream is a faithful mirror of the graph's emission order — what you see in the browser is exactly the order in which nodes wrote to state.

Past debates are persisted as `debate_<timestamp>.json` files in the working directory. The `/debates` endpoint lists them; the UI lets you replay any prior run.

### 2.7 Configurable parameters

The system is opinionated by default but exposes the parameters that change the *shape* of a debate:

- `--topic` — the question being argued.
- `--turns` — total philosopher speeches (including the opening), enforced by hard cap.
- `--moderator-every N` — how often the moderator inserts a checkpoint (`0` disables them entirely; mid-debate moderators are useful for long debates and noise for short ones).
- `--philosophers` — which subset of the roster participates. The first name in the list is the opening proponent.
- `--save` — also write the structured JSON transcript (the human-readable `.txt` is always written).
- `--rebuild-index` — force-rebuild FAISS indexes (useful after editing source texts if you somehow break the fingerprint).
- `--export-graph` — render the compiled LangGraph as a PNG.

Model and temperature are set in `config.py`: `gpt-4o-mini` for everything by default (cheap iteration), with the Director on `0.45` and the philosophers/moderator on `0.7`. Embeddings use `text-embedding-3-small`.
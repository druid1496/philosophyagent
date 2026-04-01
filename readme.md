# Philosophy Debate Agent

让不同的哲学家 agent 在当代新闻/哲学议题上辩论。

A multi-agent system where philosopher AIs debate contemporary topics — grounded in their actual texts via RAG, orchestrated by LangGraph.

---

## How It Works

1. **Philosopher Agents** — each philosopher (Plato, Kant, Nietzsche) is built from RAG over their own philosophical texts + a system prompt that keeps them in character. Every response is anchored to retrieved passages from their writings.

2. **Evaluator Agent** — after each philosopher speaks, the evaluator scores the response on:
   - *Textual Grounding*: does it reflect the philosopher's actual documented positions?
   - *Internal Consistency*: does it contradict earlier statements in this debate?
   - *Argumentative Rigor*: how philosophically sound is the reasoning?

3. **Moderator Agent** — controls the full debate lifecycle: introduces the topic, summarizes each round and poses follow-up questions, and writes a final synthesis conclusion.

4. **LangGraph** — orchestrates the multi-turn loop. Each round, all philosophers speak in sequence, each followed by an evaluation. The moderator summarizes after every full round. Runs for N configurable turns.

### Debate Flow

```
START
  └─> Moderator: introduce topic
        └─> Philosopher 1 speaks
              └─> Evaluator assesses
                    └─> Philosopher 2 speaks
                          └─> Evaluator assesses
                                └─> Philosopher 3 speaks
                                      └─> Evaluator assesses
                                            └─> Moderator: summarize round + follow-up question
                                                  └─> [repeat for N rounds]
                                                        └─> Moderator: conclusion
                                                              └─> END
```

---

## Project Structure

```
philosophyagent/
├── main.py                  # entry point — CLI with --topic, --turns, --philosophers, --save
├── config.py                # model names, philosopher registry
├── requirements.txt
├── .env.example
├── texts/
│   ├── plato.txt            # RAG source: Forms, Cave, Justice, Soul, Eros
│   ├── kant.txt             # RAG source: Categorical Imperative, Transcendental Idealism, Dignity
│   └── nietzsche.txt        # RAG source: God is Dead, Will to Power, Übermensch, Morality
├── rag/
│   └── retriever.py         # FAISS vector store per philosopher
├── agents/
│   ├── philosopher.py       # PhilosopherAgent: RAG retrieval + in-character LLM response
│   ├── evaluator.py         # EvaluatorAgent: grounding + consistency scoring
│   └── moderator.py         # ModeratorAgent: intro / turn summary / conclusion
└── graph/
    ├── state.py             # DebateState TypedDict (LangGraph state schema)
    └── debate_graph.py      # LangGraph nodes + conditional edges
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
# open .env and fill in your OPENAI_API_KEY
```

### 3. Run

```bash
# Default: Plato vs Kant vs Nietzsche, 2 rounds, AI topic
python3 main.py

# Custom topic and number of rounds
python3 main.py --topic "Is democracy the best form of government?" --turns 3

# Select specific philosophers and save transcript
python3 main.py --philosophers Plato Nietzsche --turns 2 --save
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--topic` | The debate topic | AI and human flourishing |
| `--turns` | Number of debate rounds | 2 |
| `--philosophers` | Which philosophers to include | Plato Kant Nietzsche |
| `--save` | Save full transcript to JSON | off |

---

## Adding a New Philosopher

1. Add a text file to `texts/` (e.g. `texts/hegel.txt`) with excerpts from their writings.
2. Add an entry to `PHILOSOPHERS` in `config.py`:

```python
"Hegel": {
    "text_file": "texts/hegel.txt",
    "description": "German Idealist philosopher (1770–1831), author of the Phenomenology of Spirit...",
    "era": "1770–1831",
},
```

3. Run — no other changes needed.

---

## Models

All agents use `gpt-4o-mini` by default (cheapest, good for debugging). Embeddings use `text-embedding-3-small`. Both are configurable in `config.py`:

```python
DEBATE_MODEL = "gpt-4o-mini"      # swap to "gpt-4o" for better quality
EMBEDDING_MODEL = "text-embedding-3-small"
```

---

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — multi-agent graph orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM chains, RAG, prompts
- [FAISS](https://github.com/facebookresearch/faiss) — local vector store for philosopher texts
- [OpenAI](https://platform.openai.com/) — `gpt-4o-mini` + `text-embedding-3-small`

# Notes

## Big Picture

This repo is a single Python app for running a multi-agent philosophy debate:

- each philosopher speaks using RAG over their own source text,
- a director agent decides who speaks next (or when to end),
- a moderator opens, checkpoints, and closes,
- LangGraph orchestrates the loop.

## Key Design Decisions

In graph/state.py:

- last_philosopher_speaker: “Who most recently spoke as a philosopher?”
- target_philosopher: “Who the next philosopher is supposed to engage/rebut?”
- chaos_factor is a one-turn prompt injection from the Debate Director to make the next response sharper when the debate feels too tame.
- `messages is the full canonical transcript stream as structured records (DebateMessage with turn, speaker type/name, content).
- debate_history is a compact routing log (list[str]) mainly for the director’s “friction memory” (e.g., who was matched against whom, whether chaos was injected).

How much does it keep messages and debate_history?
- In state, both are append-only via Annotated[..., operator.add], so they grow over the run.
- But downstream consumers use windowed/truncated views:
    - Director gets only the last 12 debate_history entries.
    - Director’s recent transcript view uses roughly last 10 messages (with per-message excerpting).
    - Moderator checkpoint uses transcript text trimmed to about 12000 chars.
- End-of-run output still uses the full messages list for transcript save/conclusion context.

## Improvement Ideas

- [ ] Moderator conclusion speech needs improvement.
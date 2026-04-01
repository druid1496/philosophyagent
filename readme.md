philosophy agent
让不同的哲学家 agent 在当代的新闻/哲学议题上辩论
1. 用 RAG （某个哲学家的哲学文本）+ system prompt 就可以打造该哲学家 agent，比如柏拉图，康德，尼采
2. 除了哲学家 agent，还需要有单独的 judge/evaluator agent （评估某哲学家 agent 的回答是否基于文本、是否前后矛盾，等等） 和一个 moderator agent（主持整个辩论）
  然后这个用 LangGraph 做就可以了
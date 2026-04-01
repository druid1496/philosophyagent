from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever


_PROMPT = """\
You are an impartial philosophical evaluator in a debate.

PHILOSOPHER BEING EVALUATED: {philosopher_name}
DEBATE TOPIC: {topic}

REFERENCE TEXTS FROM THIS PHILOSOPHER'S WRITINGS:
{context}

LATEST STATEMENT BY {philosopher_name}:
{latest_statement}

PREVIOUS STATEMENTS BY {philosopher_name} IN THIS DEBATE:
{previous_statements}

Evaluate the latest statement on three dimensions. Be concise but specific (total ≤ 120 words):

Textual Grounding (0–10): Does the statement reflect this philosopher's actual documented positions?
Internal Consistency (0–10): Is it consistent with the philosopher's earlier statements in this debate?
Argumentative Rigor (0–10): How philosophically sound and well-reasoned is the argument?

Format exactly as:
Textual Grounding: X/10
Internal Consistency: X/10
Argumentative Rigor: X/10
Feedback: <2-3 specific sentences>
"""


class EvaluatorAgent:
    def __init__(self, llm: ChatOpenAI, retrievers: dict[str, BaseRetriever]):
        self.retrievers = retrievers
        prompt = ChatPromptTemplate.from_messages([("human", _PROMPT)])
        self._chain = prompt | llm

    def evaluate(
        self,
        philosopher_name: str,
        topic: str,
        latest_statement: str,
        previous_statements: list[str],
    ) -> str:
        retriever = self.retrievers[philosopher_name]
        # Retrieve based on the actual statement for better grounding checks
        docs = retriever.invoke(latest_statement[:500])
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        prev = (
            "\n\n".join(f"[Turn {i+1}] {s}" for i, s in enumerate(previous_statements))
            if previous_statements
            else "None — this is the philosopher's first statement."
        )

        result = self._chain.invoke({
            "philosopher_name": philosopher_name,
            "topic": topic,
            "context": context or "(No passages retrieved.)",
            "latest_statement": latest_statement,
            "previous_statements": prev,
        })
        return result.content

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever


_SYSTEM = """\
You are {philosopher_name} ({era}).
{description}

You are participating in a multi-turn philosophical debate. Your task is to respond in character, \
drawing on your actual philosophical framework and the retrieved passages from your own writings provided below.

IMPORTANT RULES:
- Stay strictly in character as {philosopher_name}.
- Ground every claim in your genuine philosophical positions.
- Engage with what other participants have said when relevant.
- Include **one or two verbatim quotations** from the retrieved passages below: exact wording, in \
quotation marks, each quotation **one sentence** (or a single short uninterrupted clause). Do **not** \
invent or alter quoted text; only quote substrings that appear **exactly** in RETRIEVED PASSAGES. \
If there are no real passages (e.g. only a placeholder), do not fabricate quotes—say what you think \
without false citation.
- Keep your response focused and substantive (150–250 words), quotations included.
- Do NOT break character or explain your philosophy from an external viewpoint.
"""

_HUMAN = """\
DEBATE TOPIC: {topic}

RETRIEVED PASSAGES FROM YOUR WRITINGS (your verbatim quotes must come only from here):
{context}

DEBATE HISTORY SO FAR:
{history}

LATEST EVALUATION OF YOUR PREVIOUS STATEMENT:
{last_evaluation}

Now give your response as {philosopher_name}:
"""


class PhilosopherAgent:
    def __init__(
        self,
        name: str,
        description: str,
        era: str,
        retriever: BaseRetriever,
        llm: ChatOpenAI,
    ):
        self.name = name
        self.retriever = retriever
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])
        self._chain = prompt | llm

        self._base_vars = {
            "philosopher_name": name,
            "era": era,
            "description": description,
        }

    def respond(self, topic: str, history: str, last_evaluation: str) -> str:
        docs = self.retriever.invoke(topic)
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        result = self._chain.invoke({
            **self._base_vars,
            "topic": topic,
            "context": context or "(No passages retrieved.)",
            "history": history or "This is the opening statement. No prior exchanges.",
            "last_evaluation": last_evaluation or "No previous evaluation.",
        })
        return result.content

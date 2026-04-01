from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def setup_philosopher_rag(
    philosopher_name: str,
    text_file: str,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 600,
    chunk_overlap: int = 80,
    k: int = 4,
):
    """Load a philosopher's text file, chunk it, embed it into FAISS, and return a retriever."""
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    docs = [
        Document(page_content=chunk, metadata={"philosopher": philosopher_name})
        for chunk in chunks
    ]

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})

import hashlib
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTOR_STORE_ROOT = Path(__file__).resolve().parent.parent / "vector_stores"
INDEX_NAME = "index"
META_NAME = "build_meta.json"


def _fingerprint(text_file: str, embedding_model: str, chunk_size: int, chunk_overlap: int) -> str:
    path = Path(text_file)
    h = hashlib.sha256()
    h.update(path.read_bytes())
    h.update(
        f"|{embedding_model}|{chunk_size}|{chunk_overlap}".encode("utf-8"),
    )
    return h.hexdigest()


def _store_dir(philosopher_name: str) -> Path:
    safe = philosopher_name.replace(" ", "_")
    return VECTOR_STORE_ROOT / safe


def _index_ready(store_dir: Path) -> bool:
    return (store_dir / f"{INDEX_NAME}.faiss").exists() and (store_dir / f"{INDEX_NAME}.pkl").exists()


def load_philosopher_vectorstore(
    philosopher_name: str,
    text_file: str,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force_rebuild: bool = False,
) -> FAISS:
    """Load or build a persisted FAISS index for a philosopher (same cache as the debate retriever)."""
    store_dir = _store_dir(philosopher_name)
    meta_path = store_dir / META_NAME
    fp = _fingerprint(text_file, embedding_model, chunk_size, chunk_overlap)
    embeddings = OpenAIEmbeddings(model=embedding_model)

    if (
        not force_rebuild
        and _index_ready(store_dir)
        and meta_path.is_file()
    ):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
        if meta.get("fingerprint") == fp:
            return FAISS.load_local(
                str(store_dir),
                embeddings,
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True,
            )

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

    vectorstore = FAISS.from_documents(docs, embeddings)
    store_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(store_dir), index_name=INDEX_NAME)
    meta_path.write_text(
        json.dumps({"fingerprint": fp}, indent=2),
        encoding="utf-8",
    )
    return vectorstore


def setup_philosopher_rag(
    philosopher_name: str,
    text_file: str,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 4,
    force_rebuild: bool = False,
):
    """Load or build a FAISS index for a philosopher; persist under vector_stores/<name>/."""
    vectorstore = load_philosopher_vectorstore(
        philosopher_name,
        text_file,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_rebuild=force_rebuild,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})

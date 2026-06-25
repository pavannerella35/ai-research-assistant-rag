"""
ingest.py
Chunks documents in data/sample_knowledge_base and builds a TF-IDF vector index
used by the RetrieverAgent. Swappable for sentence-transformers + Chroma/FAISS
in a production deployment without changing the retriever's interface.
"""
import os
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_knowledge_base")
STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store.pkl")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                docs.append({"source": fname, "text": f.read()})
    return docs


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = (current[-overlap:] + "\n" + para) if current else para
    if current:
        chunks.append(current)
    return chunks


def build_index(data_dir: str = DATA_DIR, store_path: str = STORE_PATH) -> None:
    docs = load_documents(data_dir)
    if not docs:
        raise FileNotFoundError(f"No .txt documents found in {data_dir}")

    all_chunks, metadata = [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            all_chunks.append(chunk)
            metadata.append({"source": doc["source"], "chunk_id": i})

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    vectors = vectorizer.fit_transform(all_chunks)

    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "wb") as f:
        pickle.dump(
            {"vectorizer": vectorizer, "vectors": vectors, "chunks": all_chunks, "metadata": metadata},
            f,
        )
    print(f"Indexed {len(all_chunks)} chunks from {len(docs)} document(s) -> {store_path}")


if __name__ == "__main__":
    build_index()

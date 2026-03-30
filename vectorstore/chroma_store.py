import os
import chromadb

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
        _collection = client.get_or_create_collection("agnostic-lookup")
    return _collection


def upsert(id: str, vector: list[float], metadata: dict):
    _get_collection().upsert(
        ids=[id],
        embeddings=[vector],
        metadatas=[metadata],
        documents=[metadata["text"]],
    )


def retrieve(vector: list[float], domain: str | None = None, top_k: int = 4) -> list[dict]:
    where = {"domain": domain} if domain else None
    results = _get_collection().query(query_embeddings=[vector], n_results=top_k, where=where)
    return [
        {"name": meta["name"], "text": meta["text"]}
        for meta in results["metadatas"][0]
    ]

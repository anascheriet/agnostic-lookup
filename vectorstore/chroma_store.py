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


def retrieve(vector: list[float], domain: str | None = None, top_k: int = 4, similarity_threshold: float | None = None) -> list[dict]:
    """
    Retrieve similar documents from ChromaDB.

    Args:
        vector: Query embedding vector
        domain: Optional domain filter
        top_k: Max results to return
        similarity_threshold: Optional minimum similarity (0-1). Results below this are filtered out.
                             None = no filtering (return all top_k)

    Returns:
        List of dicts with "name", "text", and "similarity" keys.
    """
    where = {"domain": domain} if domain else None
    results = _get_collection().query(query_embeddings=[vector], n_results=top_k, where=where)

    documents = []
    distances = results.get("distances", [[]])[0]  # ChromaDB returns distances (lower = more similar)

    for i, meta in enumerate(results["metadatas"][0]):
        distance = distances[i] if i < len(distances) else 1.0
        similarity = 1 - distance  # Convert distance to similarity (0-1, higher = better)

        # Apply threshold filter
        if similarity_threshold is not None and similarity < similarity_threshold:
            continue

        documents.append({
            "name": meta["name"],
            "text": meta["text"],
            "similarity": similarity
        })

    return documents

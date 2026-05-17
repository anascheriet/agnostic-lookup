import os
from pinecone import Pinecone

_index = None


def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _index = pc.Index(os.environ["PINECONE_INDEX"])
    return _index


def upsert(id: str, vector: list[float], metadata: dict):
    _get_index().upsert(vectors=[{"id": id, "values": vector, "metadata": metadata}])


def retrieve(vector: list[float], domain: str | None = None, top_k: int = 4, similarity_threshold: float | None = None) -> list[dict]:
    filter_ = {"domain": {"$eq": domain}} if domain else None
    results = _get_index().query(vector=vector, top_k=top_k, include_metadata=True, filter=filter_)

    documents = []
    for match in results.matches:
        similarity = match.score  # Pinecone score is already 0-1

        # Apply threshold filter
        if similarity_threshold is not None and similarity < similarity_threshold:
            continue

        documents.append({
            "name": match.metadata["name"],
            "text": match.metadata["text"],
            "similarity": similarity
        })

    return documents

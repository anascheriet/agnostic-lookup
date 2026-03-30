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


def retrieve(vector: list[float], domain: str | None = None, top_k: int = 4) -> list[dict]:
    filter_ = {"domain": {"$eq": domain}} if domain else None
    results = _get_index().query(vector=vector, top_k=top_k, include_metadata=True, filter=filter_)
    return [{"name": m.metadata["name"], "text": m.metadata["text"]} for m in results.matches]

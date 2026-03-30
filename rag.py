"""
RAG core: embed a query, retrieve context from Pinecone, generate answer with Mistral.
"""

import os
from pinecone import Pinecone
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
EMBED_MODEL = "mistral-embed"
TOP_K = 4

_pinecone_index = None
_mistral_client = None


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index


def _get_mistral():
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return _mistral_client


def retrieve(query: str, domain: str | None = None, top_k: int = TOP_K) -> list[dict]:
    client = _get_mistral()
    index = _get_index()

    response = client.embeddings.create(model=EMBED_MODEL, inputs=[query])
    vector = response.data[0].embedding
    filter_ = {"domain": {"$eq": domain}} if domain else None

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_,
    )
    return results.matches


def compare(subject_a: str, subject_b: str, domain: str | None = None) -> str:
    query = f"Compare {subject_a} and {subject_b}"
    matches = retrieve(query, domain=domain)

    if not matches:
        return "No relevant information found in the knowledge base."

    context = "\n\n---\n\n".join(
        f"[{m.metadata['name']}]\n{m.metadata['text']}" for m in matches
    )

    prompt = (
        f"You are an expert analyst. Using only the context below, "
        f"compare {subject_a} and {subject_b} in depth.\n\n"
        f"Context:\n{context}\n\n"
        f"Comparison:"
    )

    client = _get_mistral()
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

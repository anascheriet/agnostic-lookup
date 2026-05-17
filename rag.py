"""
RAG core: embed a query, retrieve context from vector store, generate answer with Mistral.
"""

import os
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
EMBED_MODEL = "mistral-embed"
TOP_K = 4

_mistral_client = None


def _get_mistral():
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return _mistral_client


def _get_store():
    store = os.getenv("VECTOR_STORE", "pinecone")
    if store == "chroma":
        from vectorstore.chroma_store import retrieve
    else:
        from vectorstore.pinecone_store import retrieve
    return retrieve


def retrieve(query: str, domain: str | None = None, top_k: int = TOP_K, similarity_threshold: float | None = None) -> list[dict]:
    client = _get_mistral()
    response = client.embeddings.create(model=EMBED_MODEL, inputs=[query])
    vector = response.data[0].embedding

    # Get raw results (may include multiple chunks from same subject)
    raw_results = _get_store()(vector, domain=domain, top_k=top_k * 2, similarity_threshold=similarity_threshold)

    # Deduplicate by subject name, keeping highest-similarity chunk per subject
    seen = {}
    for result in raw_results:
        name = result["name"]
        if name not in seen:
            seen[name] = result

    # Return top_k unique subjects
    return list(seen.values())[:top_k]


def compare(subject_a: str, subject_b: str, domain: str | None = None) -> str:
    matches = retrieve(f"Compare {subject_a} and {subject_b}", domain=domain)

    if not matches:
        return "No relevant information found in the knowledge base."

    context = "\n\n---\n\n".join(f"[{m['name']}]\n{m['text']}" for m in matches)

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

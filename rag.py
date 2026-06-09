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
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))
USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"

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


def rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Use LLM to rerank results by semantic relevance to query.

    Args:
        query: Original search query
        results: List of retrieved chunks with text and name

    Returns:
        Results re-ranked by relevance (best match first)
    """
    if not results or len(results) <= 1:
        return results

    # Build candidate list for LLM to judge
    candidates_text = "\n\n".join(
        f"[{i+1}] {result['name']}: {result['text'][:200]}..."
        for i, result in enumerate(results)
    )

    prompt = (
        f"Rank these results by relevance to the query. Return ONLY the numbers in order, "
        f"highest relevance first (e.g., '2 1 3').\n\n"
        f"Query: {query}\n\n"
        f"Results:\n{candidates_text}"
    )

    client = _get_mistral()
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse ranking from response
    try:
        ranking = response.choices[0].message.content.strip().split()
        ranking_indices = [int(x) - 1 for x in ranking if x.isdigit()]

        # Reorder results based on ranking
        reranked = [results[i] for i in ranking_indices if i < len(results)]

        # Add any missing results that weren't ranked
        ranked_set = set(ranking_indices)
        for i, result in enumerate(results):
            if i not in ranked_set:
                reranked.append(result)

        return reranked
    except (ValueError, IndexError):
        return results


def retrieve(query: str, domain: str | None = None, similarity_threshold: float | None = None, use_reranking: bool | None = None) -> list[dict]:
    """
    Retrieve relevant chunks: similarity filter → dedup → optional reranking.

    Args:
        query: Search query
        domain: Optional domain filter
        similarity_threshold: Minimum similarity to include result (default: SIMILARITY_THRESHOLD)
        use_reranking: Use LLM to rerank by relevance (default: USE_RERANKING)

    Returns:
        List of chunks, optionally reranked by semantic relevance
    """
    client = _get_mistral()
    response = client.embeddings.create(model=EMBED_MODEL, inputs=[query])
    vector = response.data[0].embedding

    # Use provided threshold or fall back to default
    threshold = similarity_threshold if similarity_threshold is not None else SIMILARITY_THRESHOLD

    # Get raw results (fetch more to have options after dedup)
    raw_results = _get_store()(vector, domain=domain, top_k=MAX_RESULTS * 2, similarity_threshold=threshold)

    # Deduplicate by subject name, keeping highest-similarity chunk per subject
    seen = {}
    for result in raw_results:
        name = result["name"]
        if name not in seen:
            seen[name] = result

    results = list(seen.values())

    # Optionally rerank by semantic relevance
    if use_reranking is None:
        use_reranking = USE_RERANKING

    if use_reranking and results:
        results = rerank(query, results)

    return results


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

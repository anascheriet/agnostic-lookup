"""
Fetch Wikipedia summaries for all subjects and upsert them into the configured vector store.
Uses Mistral embeddings — no PyTorch required.
"""

import os
import time
import unicodedata
import wikipedia
from mistralai.client import Mistral
from dotenv import load_dotenv
from subjects import SUBJECTS

load_dotenv()

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
EMBED_MODEL = "mistral-embed"


def to_ascii_id(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full text to chunk
        chunk_size: Characters per chunk (default 500)
        overlap: Characters to overlap between chunks (default 100)

    Returns:
        List of text chunks

    Example:
        text = "ABCDEFGH..." (8 chars)
        chunk_size = 3, overlap = 1
        Returns: ["ABC", "BCD", "CDE", "DEF", "EFG", "FGH"]
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Move back by overlap amount

    return chunks


def get_wikipedia_text(title: str) -> list[str]:
    """
    Fetch Wikipedia article and split into chunks.
    Returns list of text chunks instead of one big string.
    """
    try:
        page = wikipedia.page(title, auto_suggest=False)
        content = page.content[:5000]
        return chunk_text(content, chunk_size=500, overlap=100)
    except Exception as e:
        print(f"  [WARN] Could not fetch '{title}': {e}")
        return []


def embed(client: Mistral, text: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, inputs=[text])
    return response.data[0].embedding


def _get_store():
    store = os.getenv("VECTOR_STORE", "pinecone")
    if store == "chroma":
        from vectorstore.chroma_store import upsert
        print("Using vector store: Chroma (local)")
    else:
        from vectorstore.pinecone_store import upsert
        print("Using vector store: Pinecone (cloud)")
    return upsert


def main():
    upsert = _get_store()

    print("Connecting to Mistral...")
    mistral = Mistral(api_key=MISTRAL_API_KEY)

    if os.getenv("VECTOR_STORE", "pinecone") == "pinecone":
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = os.environ["PINECONE_INDEX"]
        if index_name not in pc.list_indexes().names():
            print(f"Creating Pinecone index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    for domain, items in SUBJECTS.items():
        print(f"\nIngesting domain: {domain}")
        for item in items:
            print(f"  Fetching: {item}")
            chunks = get_wikipedia_text(item)
            if not chunks:
                continue

            # Store each chunk separately with a unique ID
            for chunk_idx, chunk_text in enumerate(chunks):
                vector = embed(mistral, chunk_text)
                chunk_id = f"{domain}::{to_ascii_id(item)}::chunk_{chunk_idx}"
                upsert(
                    id=chunk_id,
                    vector=vector,
                    metadata={"domain": domain, "name": item, "text": chunk_text},
                )
                time.sleep(2)  # Delay to avoid rate limiting (0.5 req/sec)

            print(f"  Upserted: {item} ({len(chunks)} chunks)")

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()

"""
Fetch Wikipedia summaries for all subjects and upsert them into the configured vector store.
Uses Mistral embeddings — no PyTorch required.
"""

import os
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


def get_wikipedia_text(title: str) -> str:
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.content[:5000]
    except Exception as e:
        print(f"  [WARN] Could not fetch '{title}': {e}")
        return ""


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
            text = get_wikipedia_text(item)
            if not text:
                continue
            vector = embed(mistral, text)
            upsert(
                id=f"{domain}::{to_ascii_id(item)}",
                vector=vector,
                metadata={"domain": domain, "name": item, "text": text},
            )
            print(f"  Upserted: {item}")

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()

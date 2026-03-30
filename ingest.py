"""
Fetch Wikipedia summaries for all subjects and upsert them into Pinecone.
Uses Mistral embeddings — no PyTorch required.
"""

import os
import unicodedata
import wikipedia
from pinecone import Pinecone, ServerlessSpec
from mistralai.client import Mistral
from dotenv import load_dotenv
from subjects import SUBJECTS

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
EMBED_MODEL = "mistral-embed"
VECTOR_DIM = 1024


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


def main():
    print("Connecting to Mistral...")
    mistral = Mistral(api_key=MISTRAL_API_KEY)

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX not in pc.list_indexes().names():
        print(f"Creating index '{PINECONE_INDEX}'...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(PINECONE_INDEX)

    for domain, items in SUBJECTS.items():
        print(f"\nIngesting domain: {domain}")
        for item in items:
            print(f"  Fetching: {item}")
            text = get_wikipedia_text(item)
            if not text:
                continue
            vector = embed(mistral, text)
            index.upsert(
                vectors=[
                    {
                        "id": f"{domain}::{to_ascii_id(item)}",
                        "values": vector,
                        "metadata": {"domain": domain, "name": item, "text": text},
                    }
                ]
            )
            print(f"  Upserted: {item}")

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()

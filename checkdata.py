import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
collection = client.get_or_create_collection("agnostic-lookup")

print(f"Total documents: {collection.count()}\n")

results = collection.get(include=["metadatas"])
for meta in results["metadatas"]:
    print(f"[{meta['domain']}] {meta['name']}")

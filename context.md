# AgnosticLookup — project context

## What this is
Python RAG app that compares football players, movies, music.
Wikipedia → Pinecone → Mistral → FastAPI.

## Current state
- ingest.py done, tested
- rag.py done
- api.py in progress

## Next steps
- add domain filter to /compare endpoint
- add movies subjects to subjects.py

## Known issues
- Pinecone index name must match .env exactly
```

Then when you start a new Claude Code session:
```
read CONTEXT.md then let's continue building AgnosticLookup

# AgnosticLookup

A RAG (Retrieval-Augmented Generation) app that compares football players, movies, and musicians using Wikipedia as a knowledge base.

## How it works

```
INGEST (run once)
  Wikipedia article → Mistral embed → Pinecone vector store

QUERY (each API call)
  user query → Mistral embed → Pinecone search → build prompt → Mistral LLM → comparison
```

Mistral is used twice: once for **embedding** (finding relevant context) and once for **generation** (writing the comparison).

## Stack

- **[Mistral](https://mistral.ai)** — embeddings (`mistral-embed`) + generation (`mistral-small`)
- **[Pinecone](https://pinecone.io)** — vector store
- **[FastAPI](https://fastapi.tiangolo.com)** — REST API
- **Wikipedia** — knowledge source

## Setup

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your keys in `.env`:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=agnostic-lookup
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_MODEL=mistral-small-latest
```

### 3. Ingest data

Fetches Wikipedia articles for all subjects and stores them in Pinecone. Run once.

```bash
python3 ingest.py
```

### 4. Start the API

```bash
python3 -m uvicorn api:app --reload
```

API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API

### `POST /compare`

Compare two subjects.

**Request:**
```json
{
  "subject_a": "Lionel Messi",
  "subject_b": "Cristiano Ronaldo",
  "domain": "football"
}
```

`domain` is optional (`football`, `movies`, `music`). Omit for cross-domain comparisons.

**Response:**
```json
{
  "subject_a": "Lionel Messi",
  "subject_b": "Cristiano Ronaldo",
  "domain": "football",
  "comparison": "..."
}
```

### `GET /domains`

List available domains.

### `GET /health`

Health check.

## Subjects

| Domain | Subjects |
|--------|----------|
| football | Lionel Messi, Cristiano Ronaldo, Kylian Mbappé, Erling Haaland, Neymar |
| movies | The Godfather, Inception, Pulp Fiction, The Dark Knight, Interstellar |
| music | Michael Jackson, The Beatles, Bob Dylan, Beyoncé, David Bowie |

To add more, edit `subjects.py` and re-run `ingest.py`.

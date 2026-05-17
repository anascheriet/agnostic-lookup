# AgnosticLookup

A RAG (Retrieval-Augmented Generation) app with an agentic layer that compares football players, movies, and musicians using Wikipedia as a knowledge base.

## Architecture

**Three interfaces:**
1. **REST API** — `/compare` endpoint for direct comparisons
2. **CLI Agent** — Multi-turn interactive agent with tool use
3. **Core RAG** — Embedding + retrieval + generation pipeline

**How it works:**

```
INGEST (run once)
  Wikipedia article → Mistral embed → ChromaDB (local vector store)

AGENT/API (each query)
  user query → Agent reasoning → Tool selection (compare/retrieve/list_domains)
  → Mistral embed → ChromaDB search → Mistral LLM → comparison response
```

Mistral is used twice: **embedding** (finding relevant context) and **generation** (writing the comparison).

## Stack

- **[Mistral](https://mistral.ai)** — embeddings (`mistral-embed`) + generation (`mistral-small-latest`)
- **[ChromaDB](https://www.trychroma.com/)** — local vector store (no setup required)
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
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_MODEL=mistral-small-latest
VECTOR_STORE=chroma
```

> Note: `VECTOR_STORE=chroma` uses local ChromaDB (no API keys needed). Set to `pinecone` to use cloud Pinecone instead.

### 3. Ingest data

Fetches Wikipedia articles for all subjects and stores them locally in ChromaDB. Run once.

```bash
python3 ingest.py
```

### 4. Run the agent (interactive CLI)

Multi-turn agent that can reason about comparisons, retrieve specific topics, and list domains.

```bash
python3 agent.py
```

Example:
```
You: Compare Messi and Ronaldo
Agent: [fetches context, generates detailed comparison]

You: Tell me more about Messi's achievements
Agent: [retrieves from knowledge base]
```

### 5. Or start the REST API

```bash
python3 -m uvicorn api:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## REST API Endpoints

### `POST /compare`

Compare two subjects using RAG.

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

## Agent Tools

The agent (`agent.py`) has access to three tools:

- **`compare(subject_a, subject_b, domain?)`** — Deep comparison using RAG
- **`retrieve(query, domain?)`** — Fetch knowledge base context for a topic
- **`list_domains()`** — Show available domains

The agent uses multi-turn reasoning to answer complex questions.

## Knowledge Base

| Domain | Subjects |
|--------|----------|
| football | Lionel Messi, Cristiano Ronaldo, Kylian Mbappé, Erling Haaland, Neymar |
| movies | The Godfather, Inception, Pulp Fiction, The Dark Knight, Interstellar |
| music | Michael Jackson, The Beatles, Bob Dylan, Beyoncé, David Bowie |

To add more subjects:
1. Edit `subjects.py`
2. Re-run `python3 ingest.py`

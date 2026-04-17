# Ivy-RAG: PDF Knowledge Assistant

> A lightweight end-to-end RAG system that ingests PDF files, retrieves evidence with hybrid semantic + keyword search, and generates grounded answers with citations — built with FastAPI and Mistral AI.

---

## Table of Contents

1. [Assignment Coverage](#1-assignment-coverage)
2. [Repository Structure](#2-repository-structure)
3. [System Architecture](#3-system-architecture)
4. [Data Ingestion](#4-data-ingestion)
5. [Query Processing](#5-query-processing)
6. [Retrieval](#6-retrieval)
7. [Post-Processing and Re-ranking](#7-post-processing-and-re-ranking)
8. [Generation](#8-generation)
9. [Hallucination Control and Refusal Logic](#9-hallucination-control-and-refusal-logic)
10. [API Reference](#10-api-reference)
11. [Frontend UI](#11-frontend-ui)
12. [Libraries and Software](#12-libraries-and-software)
13. [Security](#13-security)
14. [Scalability](#14-scalability)
15. [How to Run](#15-how-to-run)
16. [Evaluation Criteria Mapping](#16-evaluation-criteria-mapping)

---

## 1. Assignment Coverage

| Requirement | Status |
|---|---|
| FastAPI backend | ✅ |
| Mistral AI API for embeddings and generation | ✅ |
| PDF ingestion endpoint | ✅ `POST /ingest` |
| Query endpoint | ✅ `POST /query` |
| Intent detection | ✅ 6-class rule + LLM fallback |
| Query transformation | ✅ HyDE-lite dense + cleaned keyword form |
| Semantic search | ✅ Mistral embeddings + NumPy cosine |
| Keyword search + hybrid combination | ✅ Custom BM25 + Reciprocal Rank Fusion |
| Post-processing / re-ranking | ✅ MMR + neighbor expansion |
| LLM answer generation with prompt templates | ✅ Intent-switched templates |
| UI | ✅ React + Vite chat interface |
| No external RAG/search library | ✅ All retrieval implemented from scratch |
| No third-party vector database | ✅ NumPy + local files only |
| Citations / insufficient-evidence refusal | ✅ Cosine threshold gate |
| Answer shaping by intent | ✅ prose / list / table / summary templates |
| Hallucination filter | ✅ Per-sentence LLM evidence check |
| Query refusal policies | ✅ PII regex + medical/legal disclaimers |

---

## 2. Repository Structure

```
Ivy-RAG/
├── backend/
│   ├── app/
│   │   ├── ingestion/      # PDF parsing, chunking, /ingest endpoint
│   │   ├── retrieval/      # Embeddings, BM25, hybrid fusion, reranking
│   │   ├── generation/     # Prompts, Mistral chat, hallucination verifier
│   │   ├── query/          # Intent detection, policy filter, query transform
│   │   ├── storage/        # In-memory index + file persistence
│   │   ├── config.py
│   │   └── main.py
│   ├── scripts/
│   │   └── calibrate_thresholds.py
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── api/client.ts
│       └── components/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── nginx.conf
└── .env.example
```

---

## 3. System Architecture

```
User
 │
 ▼
┌─────────────────────────────────────────────────────┐
│                   React Frontend                    │
│         Upload PDFs · Chat · Citation Panel         │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI Backend                    │
│                                                     │
│  POST /ingest                  POST /query          │
│  ┌─────────────────┐          ┌──────────────────┐  │
│  │  PDF Parser     │          │  Policy Check    │  │
│  │  Chunker        │          │  Intent Detect   │  │
│  │  Embedder       │          │  Query Transform │  │
│  │  BM25 Builder   │          │  Dense Search    │  │
│  │  Index Store    │          │  BM25 Search     │  │
│  └─────────────────┘          │  RRF Fusion      │  │
│                               │  MMR Rerank      │  │
│                               │  Evidence Gate   │  │
│                               │  LLM Generation  │  │
│                               │  Halluc. Verify  │  │
│                               └──────────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌───────────────┐           ┌──────────────────┐
│  Local Index  │           │  Mistral AI API  │
│  chunks.jsonl │           │  mistral-embed   │
│  embeddings   │           │  mistral-small   │
│  bm25.pkl     │           └──────────────────┘
└───────────────┘
```

---

## 4. Data Ingestion

### Endpoint

```
POST /ingest
Content-Type: multipart/form-data
Body: files[] (one or more PDFs, max 25 MB each, max 20 files)
```

### PDF Extraction

Text is extracted page-by-page using [`pypdf`](https://pypdf.readthedocs.io/). Each chunk retains its source document name, page range, and detected section heading for citation purposes.

### Chunking Strategy

The chunker is heading-aware and paragraph-first:

1. **Section detection** — ALL-CAPS, numbered headings (`3.2 Methods`), and short title-case lines are treated as section boundaries, keeping related content together.
2. **Paragraph packing** — paragraphs are greedily packed up to ~500 tokens (~2 000 characters). Larger paragraphs are split at sentence boundaries.
3. **Overlap** — each chunk carries ~75 tokens (~300 chars) of tail from the previous chunk to prevent answers that span chunk boundaries from being missed.
4. **Metadata** — every chunk records `doc_id`, `doc_name`, `page_start`, `page_end`, `section_title`, and `chunk_index`.

**Key tradeoff:** smaller chunks improve retrieval precision; larger chunks preserve semantic context. The 500-token target is a widely-used sweet spot for embedding models. Overlap trades a small amount of storage for meaningful recall gains at boundaries.

---

## 5. Query Processing

### Intent Detection

Before touching the index, every query is classified into one of six intents using a rule layer (fast, deterministic) with a Mistral fallback for ambiguous cases:

| Intent | Behaviour |
|---|---|
| `chitchat` | Direct reply, no retrieval |
| `knowledge_lookup` | Full RAG pipeline, prose output |
| `list_request` | RAG + bulleted list template |
| `table_request` | RAG + Markdown table template |
| `summary_request` | RAG + summary template |
| `unsafe` | Immediate refusal |

This avoids unnecessary embedding calls for greetings, thanks, and off-topic queries.

### Query Transformation

Each query is transformed into two forms before retrieval:

- **Dense query (HyDE-lite):** Mistral generates a short hypothetical answer to the question; that answer text is embedded instead of the raw query. Embedding a likely-answer rather than the question produces a vector closer to the document chunks that contain the real answer.
- **Keyword query:** WH-words and filler phrases are stripped for cleaner BM25 token matching.

---

## 6. Retrieval

### Dense Search

Chunk texts are embedded at ingest time using `mistral-embed` and stored as an L2-normalized NumPy matrix. At query time:

```
similarity = embeddings_matrix @ query_vector   # cosine = dot product (both L2-normalized)
top-k selected via np.argpartition (O(N), faster than full sort)
```

No external vector database is used.

### Keyword Search — Custom BM25

A full [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) index is built from scratch:

```
score(q, d) = Σ IDF(t) × tf(t,d)×(k1+1) / (tf(t,d) + k1×(1 − b + b×|d|/avgdl))
IDF(t) = log((N − df(t) + 0.5) / (df(t) + 0.5) + 1)
k1 = 1.5,  b = 0.75
```

BM25 captures exact terms, identifiers, and acronyms that dense retrieval can miss. The index is updated incrementally on each ingest (O(new docs)) rather than rebuilt from scratch.

### Hybrid Combination — Reciprocal Rank Fusion

Dense and BM25 ranked lists are merged with RRF:

```
RRF(d) = 1/(k + rank_dense(d)) + 1/(k + rank_bm25(d))   k = 60
```

RRF is rank-based rather than score-based, so it does not require calibrating the scale difference between cosine similarities and BM25 scores.

---

## 7. Post-Processing and Re-ranking

### MMR Re-ranking

After fusion, Maximal Marginal Relevance selects a diverse set:

```
score(c) = λ · sim(c, query) − (1−λ) · max_sim(c, already_selected)   λ = 0.7
```

Bonus signals: +0.10 for heading match, +0.05 for entity overlap, +0.05 for neighbor support.

### Neighbor Expansion

The previous and next chunks of each selected result are appended. This repairs answers that were split across a chunk boundary without duplicating content in the main ranked list.

---

## 8. Generation

### Evidence Gate

Before calling the LLM, top-k chunk similarity scores are checked:

```
top-1 score ≥ 0.75  AND  avg top-3 score ≥ 0.72
```

These thresholds were calibrated empirically using `scripts/calibrate_thresholds.py`:

| Query type | top-1 mean | avg-3 mean |
|---|---|---|
| Answerable (n=5) | 0.818 | 0.812 |
| Unanswerable (n=5) | 0.704 | 0.702 |
| **Chosen threshold** | **0.75** | **0.72** |

If scores are below threshold the system returns `"Insufficient evidence in the uploaded documents."` rather than hallucinating.

### Prompt Templates

The system prompt instructs the model to answer only from provided context and cite with `[S1]`, `[S2]`, etc. A per-intent template controls output format:

| Intent | Template behaviour |
|---|---|
| `knowledge_lookup` | Prose answer with inline citations |
| `list_request` | Bulleted list, each item cited |
| `table_request` | Markdown table + key-takeaway bullets |
| `summary_request` | 2–3 sentence summary + cited evidence section |

---

## 9. Hallucination Control and Refusal Logic

### Post-hoc Sentence Verification

After generation, each sentence is embedded and compared against the retrieved chunk embeddings. Sentences whose max cosine similarity falls below a threshold (0.50) are flagged and sent to Mistral for a targeted check:

> *Does the source chunk SUPPORT, CONTRADICT, or NOT MENTION this sentence?*

- `SUPPORTED` → false alarm, sentence kept clean
- `CONTRADICTED` → highlighted red in the UI
- `NOT_MENTIONED` → highlighted amber in the UI

This catches name swaps, inverted causality, and misattributed facts — not just numerical errors.

### Query Refusal Policies

| Policy | Behaviour |
|---|---|
| PII patterns (SSN, credit card, phone, email) | Hard refusal |
| PII extraction phrases ("list all emails…") | Hard refusal |
| Medical / legal questions | Answer with mandatory disclaimer |

---

## 10. API Reference

### `POST /ingest`
Upload one or more PDF files for ingestion.

**Request:** `multipart/form-data`, field `files`

**Response:**
```json
{
  "ingested": [{"doc_id": "...", "name": "report.pdf", "n_pages": 12, "n_chunks": 47}],
  "errors": []
}
```

### `POST /query`
Ask a question against the ingested knowledge base.

**Request:**
```json
{"question": "What is the median base salary for MBAn graduates?", "top_k": 5}
```

**Response:**
```json
{
  "answer": "The median base salary was $139,496 in 2025.",
  "citations": [{"doc_name": "2025 MBAn Employment Report.pdf", "page_start": 2, "score": 0.83, "snippet": "..."}],
  "intent": "knowledge_lookup",
  "sufficient_evidence": true,
  "unsupported_sentences": []
}
```

If evidence is insufficient:
```json
{"answer": "Insufficient evidence in the uploaded documents to answer this question.", "sufficient_evidence": false}
```

### Additional Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /ready` | Reports chunks loaded |
| `GET /documents` | List ingested documents |
| `DELETE /documents/{doc_id}` | Remove a document and its chunks |

---

## 11. Frontend UI

Built with **React + Vite + TypeScript**, served via Nginx in Docker mode.

- **Sidebar:** drag-and-drop PDF upload, document list with delete buttons
- **Chat window:** intent badge per query, markdown-rendered answers (tables, bullets, bold), amber/red sentence highlighting for flagged claims
- **Citation panel:** auto-expanded source cards showing document name, page range, section, and a text snippet

---

## 12. Libraries and Software

### Backend
| Library | Purpose |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | Web API framework |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| [pypdf](https://pypdf.readthedocs.io/) | PDF text extraction |
| [NumPy](https://numpy.org/) | Vector math, cosine similarity, memory-mapped storage |
| [slowapi](https://github.com/laurentS/slowapi) | Rate limiting |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Environment variable loading |
| [httpx](https://www.python-httpx.org/) | Async HTTP client for Mistral API |
| [Mistral AI API](https://docs.mistral.ai/) | Embeddings (`mistral-embed`) and generation (`mistral-small-latest`) |

All retrieval logic (BM25, RRF, MMR, cosine search) is implemented from scratch. No external RAG orchestration library or vector database is used.

### Frontend
| Library | Purpose |
|---|---|
| [React](https://react.dev/) | UI framework |
| [Vite](https://vitejs.dev/) | Build tool and dev server |
| [react-markdown](https://github.com/remarkjs/react-markdown) + [remark-gfm](https://github.com/remarkjs/remark-gfm) | Markdown rendering with table support |

### Deployment
| Tool | Purpose |
|---|---|
| [Docker](https://www.docker.com/) + [docker-compose](https://docs.docker.com/compose/) | Containerized deployment |
| [Nginx](https://nginx.org/) | Static file serving + API proxy |

---

## 13. Security

| Measure | Implementation |
|---|---|
| File validation | Extension (`.pdf`) + MIME type + size (≤ 25 MB) + count (≤ 20) |
| Rate limiting | `slowapi` on `/ingest` and `/query` |
| CORS | Locked to `FRONTEND_ORIGIN` env var |
| PII blocking | Regex patterns for SSN, credit card, email, phone; extraction-phrase detection |
| Secret management | All keys in `.env`, never committed (`.gitignore`) |
| Optional API key auth | Set `API_KEY` in `.env`; backend requires `X-Api-Key` header on all routes (health endpoints exempt). Leave blank to disable. |

---

## 14. Scalability

The current design is intentionally lightweight — no external services, no vector database, simple local files. This satisfies the assignment constraint ("no external search library") and keeps every retrieval step transparent and directly inspectable.

**Current performance characteristics:**

| Operation | Complexity | Note |
|---|---|---|
| Dense search | O(N) | `np.argpartition`, avoids full sort |
| BM25 ingest | O(new docs) | Incremental index update, not full rebuild |
| BM25 delete | O(removed chunks) | Term-level surgical removal |
| Embedding load | Demand-paged | `np.load(..., mmap_mode="r")` — OS pages in only accessed rows |

**Known limits and growth path:**

| Current | Next step for larger scale |
|---|---|
| Exact cosine scan over full matrix | Approximate nearest neighbor (e.g. HNSW) — requires external lib, excluded by assignment constraint |
| Single-process in-memory index | Shared persistent store (PostgreSQL + pgvector, or Redis) |
| Local file persistence | Object storage (S3) for raw PDFs and index artifacts |
| No user isolation | Per-user namespacing with auth middleware |

The incremental BM25 and memory-mapped embeddings already implemented are the first steps along this path — they extend the usable corpus size without changing the architecture or adding dependencies.

---

## 15. How to Run

### Prerequisites

- Python 3.9+
- Node 18+
- A [Mistral AI API key](https://console.mistral.ai)
- Docker + docker-compose (optional)

### Option A — Local dev (hot reload)

```bash
# 1. Clone and configure
git clone https://github.com/ivyxuxt/Ivy-RAG.git
cd Ivy-RAG
cp .env.example .env
# Edit .env and set MISTRAL_API_KEY=<your-key>

# 2. Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload   # http://localhost:8000

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

### Option B — Docker (full stack)

```bash
cp .env.example .env
# Edit .env and set MISTRAL_API_KEY=<your-key>

docker compose up --build
# Frontend → http://localhost:3000
# Backend  → http://localhost:8000
```

### Run Tests

```bash
cd backend
MISTRAL_API_KEY=test FRONTEND_ORIGIN=http://localhost:5173 python -m pytest tests/ -v
```

---

## 16. Evaluation Criteria Mapping

| Criterion | How this repo addresses it |
|---|---|
| **Quality of retrieval and results** | Hybrid dense + BM25 retrieval, HyDE-lite query transform, RRF fusion, MMR reranking, neighbor expansion, calibrated evidence gate, per-sentence hallucination verification |
| **Organization and readability** | Modular layout (`ingestion/`, `retrieval/`, `generation/`, `query/`, `storage/`), config-driven constants, typed Pydantic models, 69 tests, Docker deployment |
| **Problem thinking** | Intent detection avoids unnecessary retrieval; HyDE-lite improves dense recall; RRF chosen over weighted fusion to avoid cross-scale calibration; evidence gate refuses rather than hallucinating; LLM-based verifier catches semantic errors beyond number mismatches |
| **Security** | Upload validation, CORS lockdown, rate limiting, PII blocking, secret isolation, optional API-key middleware |
| **Scalability** | O(N) argpartition search, incremental BM25 (O(new docs)), memory-mapped embeddings, clear documented growth path to ANN and shared persistence |

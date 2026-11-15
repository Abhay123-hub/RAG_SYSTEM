# Multi-PDF Question Answering

Streamlit app for asking questions about uploaded PDFs. A LangGraph router directs each message to document retrieval, general chat, or a polite out-of-scope response. Document questions use page-aware chunking, OpenAI embeddings, session-isolated Chroma storage, BM25 plus vector search, reciprocal-rank fusion, reranking, relevance grading, and cited answers.

## Setup

Requires Python 3.11 or newer and an OpenAI API key.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start the app:

```powershell
streamlit run app.py
```

The app stores uploaded PDFs, Chroma data, application logs, and per-request JSONL traces under `data/` and `logs/`. These runtime files are ignored by Git. Each browser session gets its own Chroma collection and graph memory. Use the sidebar to upload/process PDFs, select files for retrieval, delete documents, or clear the chat.

## Configuration

All settings are read from `.env`. Defaults are in `.env.example`: model names, chunk size/overlap, retrieval and rerank counts, relevance and router-confidence thresholds, input length, history window, log level, and storage directories. `CHUNK_OVERLAP` must be smaller than `CHUNK_SIZE`.

## Architecture

`app.py` builds and caches session runtime services. `src/ingestion/` extracts page text, splits pages independently, hashes files, and indexes chunks. `src/retrieval/` combines semantic and BM25 rankings using reciprocal-rank fusion, then reranks and grades the candidates. `src/agents/` routes and executes the workflow as a conditional LangGraph with an in-memory checkpointer. `src/generation/` rewrites follow-ups, generates context-grounded answers, and validates inline citations. `src/ui/` contains the Streamlit upload, chat, source, and trace views.

Scanned PDFs are reported because OCR is not included. The graph checkpointer is in-memory and does not preserve chat history after an app restart. Retrieval data remains persistent in Chroma. LangSmith and Docker are not part of this setup.

## Logs and evaluation

`logs/app.log` contains structured JSON application logs, `logs/error.log` contains warnings and errors, and `logs/traces.jsonl` stores per-request summaries. To evaluate, populate `src/evaluation/dataset.json` with questions, expected answers, expected source pages, and an `unanswerable` flag, then run:

```powershell
python -m src.evaluation.run_eval
```

Each dataset entry has `question`, `expected_answer`, `expected_sources` (a list of `{ "file_name": "...", "page": 1 }` objects), and `unanswerable`. The runner indexes PDFs under `data/uploads/` before evaluation, so upload the evaluation corpus through the app first. It reports retrieval hit rate for examples with expected sources, plus answer score, faithfulness, and not-found accuracy. The repository starts with an empty dataset because expected answers and source pages must come from the PDFs being evaluated.

## Tests

```powershell
pytest
```

Unit tests use fake model and storage interfaces and do not make OpenAI API calls.
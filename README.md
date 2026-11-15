# PDF RAG Assistant

A Streamlit application for asking questions about uploaded PDF documents. It combines local document embeddings, hybrid search, and an LLM to produce answers grounded in the selected PDFs with page-level citations.

## Features

- Upload and process multiple PDFs.
- Extract text page by page and split it into overlapping chunks.
- Generate embeddings locally with Hugging Face Sentence Transformers.
- Store vectors in a Chroma collection isolated to each browser session.
- Combine semantic search and BM25 keyword search, then merge results with reciprocal-rank fusion.
- Rewrite follow-up questions, rerank passages, and check whether retrieved text supports an answer.
- Route requests to document search, general chat, or an out-of-scope response.
- Generate answers with inline citations to source PDF pages.
- Stream generated answer tokens in the chat UI.

## How it works

1. Upload PDFs and select **Process PDFs**.
2. The app extracts text and page numbers, then splits each page into chunks.
3. Sentence Transformers creates local embeddings for the chunks. The first run downloads the configured model from Hugging Face; subsequent embedding and search operations run locally.
4. Chroma stores the vectors and metadata. BM25 indexes the same session's text for keyword search.
5. For a document question, the app rewrites it if needed, retrieves and combines candidates from both search methods, reranks them, and checks their relevance.
6. The configured chat model generates an answer from the retrieved passages. The app returns citations using the source file and page number.

General questions can be answered without document retrieval. Requests outside the supported scope receive a polite refusal.

## Requirements

- Python 3.11 or newer.
- Internet access to install dependencies and download the embedding model on first use.
- An API key for the selected chat provider:
  - Groq (`GROQ_API_KEY`), or
  - OpenAI (`OPENAI_API_KEY`).

No API key is needed for embeddings. The embedding model is downloaded from Hugging Face and runs locally. Chat requests, including retrieved document passages used to answer questions, are sent to the configured LLM provider.

## Quick start (Windows PowerShell)

From the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your Groq API key:

```dotenv
GROQ_API_KEY=your-groq-api-key
LLM_PROVIDER=groq
```

The example configuration uses `openai/gpt-oss-120b` for answers and `openai/gpt-oss-20b` for routing and other compact LLM tasks. These are Groq model IDs; the `openai/` prefix is part of the model name and does not mean requests go to OpenAI. Choose model IDs available to your Groq account. The router model should support the structured outputs used by the app.

Start the application:

```powershell
streamlit run app.py
```

The first launch may take longer while the embedding model is downloaded. Restart Streamlit after changing `.env`. Environment variables set outside `.env` can override values in that file.

### Using OpenAI for chat instead

Set `LLM_PROVIDER=openai` and add `OPENAI_API_KEY` to `.env`. The same OpenAI key is used for chat only; embeddings remain local in either provider configuration.

## Configuration

All settings can be supplied in `.env`. See `.env.example` for the working Groq/local-embedding configuration.

| Setting | Purpose |
| --- | --- |
| `LLM_PROVIDER` | Chat provider: `groq` or `openai`. |
| `GROQ_API_KEY` | API key used when `LLM_PROVIDER=groq`. |
| `OPENAI_API_KEY` | API key used when `LLM_PROVIDER=openai`. Not needed for embeddings. |
| `LLM_MODEL` | Model used for answer generation and general chat. |
| `ROUTER_MODEL` | Model used for routing, query rewriting, reranking, relevance grading, and evaluation judging. |
| `EMBEDDING_MODEL` | Hugging Face Sentence Transformers model ID. Defaults to `sentence-transformers/all-MiniLM-L6-v2`. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Text chunk length and overlap. Overlap must be smaller than chunk size. |
| `RETRIEVAL_K` | Number of hybrid retrieval candidates. |
| `RERANK_TOP_N` | Maximum number of passages retained after reranking. |
| `SCORE_THRESHOLD` | Minimum relevance score for retrieved context. |
| `ROUTER_CONFIDENCE_THRESHOLD` | Minimum confidence for a non-default route. |
| `MAX_INPUT_LENGTH` | Maximum accepted user input length. |
| `HISTORY_TURNS` | Number of conversation turns retained for context. |
| `LOG_LEVEL` | Application logging level. |
| `UPLOAD_DIR` | Location for uploaded source PDFs. |
| `CHROMA_DIR` | Persistent Chroma database directory. |
| `LOG_DIR` | Application log and request trace directory. |

If `LLM_PROVIDER` is omitted, the application defaults to OpenAI. A configured process-level `LLM_PROVIDER` environment variable takes precedence over `.env`.

Changing `EMBEDDING_MODEL` creates a separate Chroma collection so vectors from different models are not mixed. Existing indexed documents are not automatically re-embedded; process the PDFs again to add them to the new collection.

## Using the application

- **Upload and process:** Use the sidebar uploader, then select **Process PDFs**. Scanned/image-only pages are not OCR'd.
- **Limit search to specific files:** Select one or more files under **Search in**.
- **Delete a document:** The **Delete** button removes its indexed passages from the active Chroma collection. It does **not** delete the original PDF from `data/uploads/<session-id>/`; remove that file separately if you also want to delete the stored source.
- **Clear chat:** Resets the current conversation thread; it does not delete uploaded PDFs or their index.

The `data/` and `logs/` directories are runtime data and are ignored by Git. Each browser session gets its own upload directory, vector collection, and in-memory graph state. Conversation history does not survive an application restart, while uploaded PDFs and Chroma data remain on disk.

## Project layout

```text
app.py                  Streamlit entry point and session runtime
config/settings.py      Environment-backed application settings
src/agents/             LangGraph workflow and routing
src/embeddings/         Local Sentence Transformers adapter
src/evaluation/         Evaluation dataset, metrics, and runner
src/generation/         Query rewriting, answer generation, citations
src/ingestion/          PDF loading, chunking, and file registry
src/retrieval/          Hybrid search, reranking, relevance grading
src/ui/                 Chat, sidebar, sources, and debug trace
src/vectorstore/        Chroma vector-store implementation
tests/                  Unit tests
```

## Logs and evaluation

Logs and request traces are written under `LOG_DIR` (by default, `logs/`):

- `app.log` contains structured application events.
- `error.log` contains warnings and errors.
- `traces.jsonl` contains per-request summaries.

To run the evaluation workflow:

1. Add PDFs to the app so they are available under `data/uploads/`.
2. Populate `src/evaluation/dataset.json`.
3. Run:

   ```powershell
   python -m src.evaluation.run_eval
   ```

Each dataset entry contains:

```json
{
  "question": "What does the report say?",
  "expected_answer": "A short reference answer",
  "expected_sources": [
    { "file_name": "report.pdf", "page": 1 }
  ],
  "unanswerable": false
}
```

The runner reports retrieval hit rate, answer score, faithfulness, and not-found accuracy. The repository's dataset is empty by default because expected answers and source pages depend on the PDFs being evaluated.

## Tests

Run the unit tests with:

```powershell
python -m pytest
```

Tests use fake model, embedding, and storage interfaces; they do not call the LLM provider or download the embedding model.

## Limitations

- OCR is not included; image-only/scanned PDF pages do not provide searchable text.
- Conversation state is held in memory and is lost when the app restarts.
- Deleting a PDF in the UI removes its vector index, not its uploaded source file.
- Changing the embedding model requires reprocessing PDFs to populate the new model-specific collection.

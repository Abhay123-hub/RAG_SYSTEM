# Architecture

## Runtime overview

The Multi-Agent RAG Platform is a session-oriented Streamlit application. The session
runtime owns a document collection, in-memory LangGraph state, and a file registry.

```text
Streamlit UI
  -> RAGNodes / LangGraph
     -> Safety and router
     -> Query rewriter
     -> Hybrid retriever
     -> Reranker and relevance grader
     -> Answer generator
     -> Citation builder
  -> Chroma vector store
  -> Chat provider
```

## Shared contracts

`src/contracts/schemas.py` defines request and response payloads. `src/contracts/events.py`
defines stream events and `src/contracts/errors.py` defines typed API errors.
`src/contracts/engine.py` defines the `RAGEngine` protocol that isolates API behavior
from LangGraph and retrieval implementation.

Consumers must validate payloads with these models and must not introduce equivalent
client-only schemas.

## Document lifecycle

1. A multipart upload is normalized to `file_name` and `content`.
2. PDF bytes are validated and text is extracted page by page.
3. Content is split into chunks that preserve file name, page, and SHA-256 hash.
4. Chunks are added to the session Chroma collection.
5. Listing derives document counts from collection metadata.
6. Deletion removes collection chunks for the selected document.

The SHA-256 hash is the content identity. Two files with the same filename but different
bytes are distinct documents. Identical bytes are duplicates and are not re-indexed.

## Retrieval

The hybrid retriever combines vector similarity with BM25 keyword ranking using
reciprocal-rank fusion. Optional `document_ids` or file-name filters limit candidates
to selected PDFs. The reranker reduces the candidate set, and the relevance grader
decides whether the context is sufficient.

Routes are `rag_query`, `general_chat`, and `out_of_scope`. Insufficient document context
returns a not-found answer and no citations.

## Answer and citation flow

The answer generator receives only retrieved context and produces an answer. The citation
builder validates citations against retrieved chunks and emits page-level source snippets.
Unverified citations are excluded.

## Streaming

SSE streams emit `token`, `citations`, `done`, and `error` events. A stream begins with
zero or more token events, may emit citations, and ends with exactly one terminal
`done` or `error` event. Consumers must allow providers to fail mid-stream and must
report the terminal error explicitly.

## Testing boundaries

Unit tests mock embeddings, chat models, vector stores, and engine boundaries.
Integration tests compose contract models with mock RAG engines. PDF tests use generated
PyMuPDF documents so no external model or network is required.

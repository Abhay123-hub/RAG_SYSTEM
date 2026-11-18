# API Contract

The API contract is implemented by Pydantic models under `src/contracts/`.
An API request should use UUID request IDs and a session ID. The default session
identifier is `local`.

## Common error response

```json
{
  "error": {
    "code": "document_not_found",
    "message": "Document was not found."
  },
  "request_id": "12345678-1234-5678-1234-567812345678"
}
```

Error codes are restricted by `ErrorCode`, including `invalid_request`,
`session_not_found`, `document_not_found`, `upload_too_large`,
`unsupported_media_type`, `validation_error`, `invalid_pdf`,
`no_extractable_text`, `upstream_model_error`, `service_unavailable`, and
`internal_error`.

## Health and session

- `GET /health`: `{ "status": "ok" }`
- `POST /session`: `{ "session_id": "local" }`
- `POST` endpoints accept no fields with `EmptyRequest` where applicable.

## Upload document

The HTTP upload field is named `file`. The multipart filename and bytes are normalized
to `UploadDocumentRequest`.

```text
file_name: string, required, 1-255 characters, ends with .pdf (case-insensitive)
content: bytes, required, 1 byte through 50 MiB
```

Success:

```json
{
  "request_id": "12345678-1234-5678-1234-567812345678",
  "document": {
    "document_id": "64 lowercase hexadecimal characters",
    "file_name": "report.pdf",
    "indexed_page_count": 3,
    "chunk_count": 8
  },
  "duplicate": false
}
```

The `document_id` is the SHA-256 content hash. `duplicate` is true only when identical
content was already indexed.

## List documents

```json
{
  "request_id": "12345678-1234-5678-1234-567812345678",
  "documents": [
    {
      "document_id": "64 lowercase hexadecimal characters",
      "file_name": "report.pdf",
      "indexed_page_count": 3,
      "chunk_count": 8
    }
  ]
}
```

## Delete document

Delete by the 64-character document ID. The document ID must exist in the current
session or the API returns `document_not_found`.

## Query

```json
{
  "query": "What changed in the report?",
  "conversation_id": "22345678-1234-5678-1234-567812345678",
  "document_ids": ["64 lowercase hexadecimal characters"],
  "include_trace": false
}
```

- `query` is stripped and must contain 1-10,000 characters.
- `conversation_id` is optional.
- `document_ids` is optional, non-empty, and unique.
- Unknown or cross-session document IDs produce `document_not_found`.
- `include_trace` requests diagnostic trace data.

Success:

```json
{
  "request_id": "12345678-1234-5678-1234-567812345678",
  "conversation_id": "22345678-1234-5678-1234-567812345678",
  "status": "answered",
  "route": "rag_query",
  "answer": "The report states the result [report.pdf, p.2].",
  "citations": [
    {
      "document_id": "64 lowercase hexadecimal characters",
      "file_name": "report.pdf",
      "page": 2,
      "snippet": "The result...",
      "score": 0.91
    }
  ],
  "trace": null
}
```

Allowed statuses are `answered`, `insufficient_context`, and `refused`. Allowed routes
are `rag_query`, `general_chat`, and `out_of_scope`.

## Server-sent events

The stream is terminated by `done` or `error`.

### Token

```text
event: token
data: {"request_id":"...","delta":"The "}
```

### Done

```text
event: done
data: {"request_id":"...","conversation_id":"...","status":"answered","route":"rag_query","answer":"...","citations":[],"trace":null}
```

### Error

```text
event: error
data: {"error":{"code":"upstream_model_error","message":"The model provider failed."},"request_id":"..."}
```

An interrupted stream must end with an `error` event rather than an unannounced close.

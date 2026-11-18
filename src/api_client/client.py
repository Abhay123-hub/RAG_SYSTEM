import json
import uuid
from collections.abc import Iterator
from typing import Any
from urllib import error, request

from pydantic import ValidationError

from src.contracts.errors import ErrorResponse
from src.contracts.schemas import (
    CreateSessionResponse,
    IngestDocumentResponse,
    ListDocumentsResponse,
    QueryResponse,
)


class APIError(Exception):
    def __init__(self, message: str, code: str = "service_unavailable", request_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.request_id = request_id


class APIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: str | None = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _session_path(self, path: str) -> str:
        if self.session_id is None:
            raise APIError(
                "Initialize a session before making this request.",
                "invalid_request",
            )
        return f"/api/sessions/{self.session_id}{path}"

    def _read_error(self, http_error: error.HTTPError) -> APIError:
        try:
            payload = json.loads(http_error.read().decode("utf-8"))
            response = ErrorResponse.model_validate(payload)
            detail = response.error
            return APIError(detail.message, detail.code, str(response.request_id))
        except (json.JSONDecodeError, ValidationError, UnicodeDecodeError):
            return APIError(
                f"Backend returned HTTP {http_error.code}: {http_error.reason}",
                "service_unavailable",
            )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        multipart: tuple[str, bytes] | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {"Accept": "application/json"}
        body: bytes | None = None

        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif multipart is not None:
            file_name, content = multipart
            boundary = f"----streamlit-{uuid.uuid4().hex}"
            safe_name = file_name.replace("\\", "_").replace('"', "_").replace("\r", "").replace("\n", "")
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
                "Content-Type: application/pdf\r\n"
                "\r\n"
            ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

        req = request.Request(self._url(path), data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read()
        except error.HTTPError as http_error:
            raise self._read_error(http_error) from http_error
        except error.URLError as url_error:
            raise APIError(f"Could not reach backend: {url_error.reason}") from url_error

        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as decode_error:
            raise APIError("Backend returned invalid JSON.") from decode_error

    def ensure_session(self) -> CreateSessionResponse:
        payload = self._request("POST", "/api/sessions")
        session = CreateSessionResponse.model_validate(payload)
        self.session_id = session.session_id
        return session

    def health(self) -> dict[str, Any]:
        payload = self._request("GET", "/health")
        return payload

    def list_documents(self) -> ListDocumentsResponse:
        payload = self._request("GET", self._session_path("/documents"))
        return ListDocumentsResponse.model_validate(payload)

    def upload_document(self, file_name: str, content: bytes) -> IngestDocumentResponse:
        payload = self._request(
            "POST",
            self._session_path("/documents"),
            multipart=(file_name, content),
        )
        return IngestDocumentResponse.model_validate(payload)

    def delete_document(self, document_id: str) -> None:
        self._request(
            "DELETE",
            self._session_path(f"/documents/{document_id}"),
        )

    def query(
        self,
        *,
        query: str,
        conversation_id: str,
        document_ids: list[str] | None = None,
        include_trace: bool = False,
    ) -> QueryResponse:
        payload = self._request(
            "POST",
            self._session_path("/query"),
            json_body={
                "query": query,
                "conversation_id": conversation_id,
                "document_ids": document_ids,
                "include_trace": include_trace,
            },
        )
        return QueryResponse.model_validate(payload)

    def stream_query(
        self,
        *,
        query: str,
        conversation_id: str,
        document_ids: list[str] | None = None,
        include_trace: bool = False,
    ) -> Iterator[dict[str, Any]]:
        body = json.dumps(
            {
                "query": query,
                "conversation_id": conversation_id,
                "document_ids": document_ids,
                "include_trace": include_trace,
            }
        ).encode("utf-8")
        req = request.Request(
            self._url(self._session_path("/query/stream")),
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        try:
            response = request.urlopen(req, timeout=self.timeout)
        except error.HTTPError as http_error:
            raise self._read_error(http_error) from http_error
        except error.URLError as url_error:
            raise APIError(f"Could not reach backend: {url_error.reason}") from url_error

        event_name = "message"
        data_lines: list[str] = []
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if data_lines:
                        data = "\n".join(data_lines)
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            payload = {"delta": data}
                        yield {"event": event_name, "data": payload}
                    event_name = "message"
                    data_lines = []
                elif line.startswith(":"):
                    continue
                elif line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].lstrip())

class RAGSystemError(Exception):
    """Base exception for errors safe to categorize at application boundaries."""


class IngestionError(RAGSystemError):
    """Raised when a document cannot be loaded or indexed."""


class RetrievalError(RAGSystemError):
    """Raised when document retrieval fails."""


class LLMError(RAGSystemError):
    """Raised when a model request fails."""


class RoutingError(RAGSystemError):
    """Raised when a router response cannot be interpreted."""


class SessionNotFoundError(RAGSystemError):
    """Raised when an operation refers to an unknown session."""


class DocumentNotFoundError(RAGSystemError):
    """Raised when an operation refers to an unknown document."""


class InvalidPDFError(RAGSystemError):
    """Raised when a file is not a readable PDF."""


class NoExtractableTextError(RAGSystemError):
    """Raised when a PDF contains no selectable text."""
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class PageContent(BaseModel):
    file_name: str
    page: int = Field(ge=1)
    text: str
    file_hash: str = ""


class Chunk(BaseModel):
    text: str
    file_name: str
    page: int = Field(ge=1)
    chunk_id: str
    file_hash: str
    score: float | None = None

    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "file_name": self.file_name,
                "page": self.page,
                "chunk_id": self.chunk_id,
                "file_hash": self.file_hash,
            },
        )

    @classmethod
    def from_document(cls, document: Document, score: float | None = None) -> "Chunk":
        return cls(
            text=document.page_content,
            file_name=str(document.metadata["file_name"]),
            page=int(document.metadata["page"]),
            chunk_id=str(document.metadata["chunk_id"]),
            file_hash=str(document.metadata["file_hash"]),
            score=score,
        )


class Source(BaseModel):
    file_name: str
    page: int
    snippet: str
    score: float | None = None


class RAGResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    request_id: str
    route: str
    trace: dict[str, object] = Field(default_factory=dict)
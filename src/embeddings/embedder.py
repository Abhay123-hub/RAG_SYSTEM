from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings


class EmbeddingService:
    """Local Sentence Transformers embedding adapter."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)
from openai import RateLimitError
from langchain_openai import OpenAIEmbeddings

from config.settings import get_settings
from src.core.decorators import retry


class EmbeddingService:
    """OpenAI embedding adapter with rate-limit retries."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key.get_secret_value())

    @retry(attempts=3, exceptions=(RateLimitError,))
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)

    @retry(attempts=3, exceptions=(RateLimitError,))
    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)
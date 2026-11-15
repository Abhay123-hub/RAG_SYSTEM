from config.settings import Settings
from src.embeddings import embedder


def test_embedding_service_uses_configured_huggingface_model(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeEmbeddings:
        def __init__(self, model_name: str) -> None:
            captured["model_name"] = model_name

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    monkeypatch.setattr(embedder, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(
        embedder,
        "get_settings",
        lambda: Settings(_env_file=None, embedding_model="test/sentence-transformer"),
    )

    service = embedder.EmbeddingService()

    assert captured["model_name"] == "test/sentence-transformer"
    assert service.embed_documents(["text"]) == [[1.0]]
    assert service.embed_query("text") == [1.0]

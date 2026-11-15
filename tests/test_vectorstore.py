from config.settings import Settings
from src.vectorstore import chroma_store


def test_embedding_model_change_uses_a_separate_collection(tmp_path, monkeypatch) -> None:
    collection_names: list[str] = []

    def fake_chroma(**kwargs):
        collection_names.append(kwargs["collection_name"])
        return object()

    monkeypatch.setattr(chroma_store, "Chroma", fake_chroma)
    settings = Settings(
        _env_file=None,
        chroma_dir=tmp_path,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    monkeypatch.setattr(chroma_store, "get_settings", lambda: settings)
    chroma_store.ChromaVectorStore("session", object())

    settings.embedding_model = "another/model"
    chroma_store.ChromaVectorStore("session", object())

    assert collection_names[0] != collection_names[1]

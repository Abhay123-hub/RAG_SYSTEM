from pydantic import SecretStr

from config.settings import Settings
from src.core import llm


def test_groq_chat_model_uses_groq_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(llm, "ChatOpenAI", fake_chat_openai)
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=SecretStr("test-groq-key"),
    )

    llm.create_chat_model("groq-test", settings=settings)

    assert captured["model"] == "groq-test"
    assert captured["api_key"] == "test-groq-key"
    assert captured["base_url"] == "https://api.groq.com/openai/v1"


def test_groq_chat_model_requires_groq_key() -> None:
    settings = Settings(_env_file=None, llm_provider="groq")

    try:
        llm.create_chat_model("groq-test", settings=settings)
    except ValueError as error:
        assert "GROQ_API_KEY" in str(error)
    else:
        raise AssertionError("Expected missing Groq key to be reported")

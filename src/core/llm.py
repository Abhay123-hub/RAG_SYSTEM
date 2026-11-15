from langchain_openai import ChatOpenAI

from config.settings import Settings, get_settings


def create_chat_model(
    model: str,
    *,
    temperature: float = 0,
    streaming: bool = False,
    settings: Settings | None = None,
) -> ChatOpenAI:
    settings = settings or get_settings()
    if settings.llm_provider == "groq":
        api_key = settings.groq_api_key.get_secret_value()
        base_url = "https://api.groq.com/openai/v1"
        key_name = "GROQ_API_KEY"
    else:
        api_key = settings.openai_api_key.get_secret_value()
        base_url = None
        key_name = "OPENAI_API_KEY"

    if not api_key:
        raise ValueError(f"Set {key_name} to use the configured LLM provider.")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
        api_key=api_key,
        base_url=base_url,
    )

from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_settings
from src.core.llm import create_chat_model
from src.generation.prompts import ANSWER_SYSTEM_PROMPT
from src.models.schemas import Chunk


class AnswerGenerator:
    """Generate grounded answers and optionally forward tokens to a UI callback."""

    def __init__(self, model: object | None = None) -> None:
        settings = get_settings()
        self.model = model or create_chat_model(settings.llm_model, streaming=True)

    def generate(self, question: str, chunks: list[Chunk], on_token: Callable[[str], None] | None = None) -> str:
        context = "\n\n".join(
            f"[{chunk.file_name}, p.{chunk.page}]\n{chunk.text}" for chunk in chunks
        )
        messages = [
            SystemMessage(content=f"{ANSWER_SYSTEM_PROMPT}\n\nDocument context:\n{context}"),
            HumanMessage(content=question),
        ]
        if on_token is None:
            result = self.model.invoke(messages)
            return str(result.content)

        tokens: list[str] = []
        for part in self.model.stream(messages):
            token = str(part.content)
            if token:
                tokens.append(token)
                on_token(token)
        return "".join(tokens)
from openai import RateLimitError
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from src.core.decorators import retry
from src.generation.prompts import REWRITE_PROMPT
from src.memory.history_manager import HistoryManager


class QueryRewriter:
    def __init__(self, model: object | None = None, history_manager: HistoryManager | None = None) -> None:
        settings = get_settings()
        self.model = model or ChatOpenAI(model=settings.router_model, temperature=0, api_key=settings.openai_api_key.get_secret_value())
        self.history_manager = history_manager or HistoryManager(settings.history_turns)
        self.prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)

    @retry(attempts=3, exceptions=(RateLimitError,))
    def rewrite(self, question: str, messages: list[object]) -> str:
        history = self.history_manager.format(messages)
        result = (self.prompt | self.model).invoke({"question": question, "history": history})
        return str(result.content).strip() or question
from pydantic import BaseModel, Field
from openai import RateLimitError
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from src.core.decorators import retry
from src.core.logger import get_logger, log_event
from src.models.schemas import Chunk

logger = get_logger(__name__)


class RankedItem(BaseModel):
    index: int = Field(ge=0)
    score: float = Field(ge=0, le=10)


class RankingResult(BaseModel):
    items: list[RankedItem]


class Reranker:
    """Use a compact structured LLM judgment to order hybrid candidates."""

    def __init__(self, model: object | None = None, top_n: int | None = None) -> None:
        settings = get_settings()
        self.model = model or ChatOpenAI(model=settings.router_model, temperature=0, api_key=settings.openai_api_key.get_secret_value()).with_structured_output(RankingResult)
        self.top_n = top_n or settings.rerank_top_n

    @retry(attempts=3, exceptions=(RateLimitError,))
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Score each passage for answering the question. Return every supplied zero-based index once, with a relevance score from 0 to 10."),
                ("human", "Question: {question}\nPassages:\n{passages}"),
            ]
        )
        passages = "\n\n".join(f"[{index}] {chunk.text}" for index, chunk in enumerate(chunks))
        try:
            result: RankingResult = (prompt | self.model).invoke({"question": query, "passages": passages})
            score_map = {item.index: item.score for item in result.items if item.index < len(chunks)}
            ranked = sorted(enumerate(chunks), key=lambda item: score_map.get(item[0], -1), reverse=True)
            output = [chunk.model_copy(update={"score": score_map[index] / 10}) for index, chunk in ranked[: self.top_n] if index in score_map]
            if not output:
                return chunks[: self.top_n]
            log_event(logger, "reranking_done", data={"before": len(chunks), "after": len(output)})
            return output
        except RateLimitError:
            raise
        except Exception:
            logger.exception("reranking_failed", extra={"event": "reranking_failed"})
            return chunks[: self.top_n]
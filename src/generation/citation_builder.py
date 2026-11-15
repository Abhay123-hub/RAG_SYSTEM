import re

from src.core.logger import get_logger, log_event
from src.models.schemas import Chunk, Source

logger = get_logger(__name__)


class CitationBuilder:
    """Build source snippets only for citations present in retrieved context."""

    _pattern = re.compile(r"\[([^\[\]]+),\s*p\.(\d+)\]", re.IGNORECASE)

    def build(self, answer: str, chunks: list[Chunk]) -> tuple[list[Source], list[str]]:
        available: dict[tuple[str, int], Chunk] = {}
        for chunk in chunks:
            available.setdefault((chunk.file_name.casefold(), chunk.page), chunk)
        sources: list[Source] = []
        invalid: list[str] = []
        used: set[tuple[str, int]] = set()
        for file_name, page_text in self._pattern.findall(answer):
            key = (file_name.strip().casefold(), int(page_text))
            chunk = available.get(key)
            if chunk is None:
                invalid.append(f"{file_name}, p.{page_text}")
                continue
            if key not in used:
                used.add(key)
                sources.append(Source(file_name=chunk.file_name, page=chunk.page, snippet=chunk.text[:240], score=chunk.score))
        if invalid:
            log_event(logger, "invalid_citations", data={"citations": invalid})
        return sources, invalid
from src.generation.citation_builder import CitationBuilder
from src.models.schemas import Chunk


def test_citation_builder_validates_retrieved_sources() -> None:
    chunks = [Chunk(text="Relevant passage", file_name="guide.pdf", page=2, chunk_id="1", file_hash="hash")]

    sources, invalid = CitationBuilder().build("Supported [guide.pdf, p.2], unsupported [other.pdf, p.9].", chunks)

    assert [(source.file_name, source.page) for source in sources] == [("guide.pdf", 2)]
    assert invalid == ["other.pdf, p.9"]
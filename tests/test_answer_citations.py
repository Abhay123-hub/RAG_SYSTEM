from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_builder import CitationBuilder
from src.models.schemas import Chunk


class FakeResult:
    content = "The forecast is stable [forecast.pdf, p.3]."


class FakeChatModel:
    def invoke(self, messages):
        assert "forecast.pdf, p.3" not in str(messages[-1].content)
        return FakeResult()

    def stream(self, messages):
        for delta in ("The ", "forecast ", "is stable."):
            yield type("Part", (), {"content": delta})()


def test_answer_generation_receives_grounding_context() -> None:
    model = FakeChatModel()
    generator = AnswerGenerator(model=model)
    chunk = Chunk(
        text="The forecast remains stable.",
        file_name="forecast.pdf",
        page=3,
        chunk_id="forecast-1",
        file_hash="hash",
    )

    answer = generator.generate("What is the forecast?", [chunk])

    assert "forecast" in answer.lower()


def test_streaming_answer_reports_tokens_and_combines_output() -> None:
    generator = AnswerGenerator(model=FakeChatModel())
    chunk = Chunk(text="Stable.", file_name="forecast.pdf", page=3, chunk_id="1", file_hash="hash")
    tokens = []

    answer = generator.generate("Forecast?", [chunk], on_token=tokens.append)

    assert answer == "The forecast is stable."
    assert tokens == ["The ", "forecast ", "is stable."]


def test_citations_include_only_sources_present_in_retrieval_context() -> None:
    chunk = Chunk(
        text="The forecast remains stable.",
        file_name="forecast.pdf",
        page=3,
        chunk_id="1",
        file_hash="hash",
        score=0.87,
    )
    answer = "Stable [forecast.pdf, p.3], but not [missing.pdf, p.9]."

    sources, invalid = CitationBuilder().build(answer, [chunk])

    assert [(source.file_name, source.page, source.score) for source in sources] == [
        ("forecast.pdf", 3, 0.87)
    ]
    assert invalid == ["missing.pdf, p.9"]

import json
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app import get_runtime
from config.settings import get_settings
from src.evaluation.metrics import mean_score, not_found_accuracy, retrieval_hit_rate


class JudgeScores(BaseModel):
    answer_score: float = Field(ge=0, le=1)
    faithfulness_score: float = Field(ge=0, le=1)


def main() -> None:
    dataset_path = Path(__file__).with_name("dataset.json")
    examples = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not examples:
        print("No evaluation examples configured in src/evaluation/dataset.json.")
        return

    if not get_settings().openai_api_key.get_secret_value():
        raise SystemExit("Set OPENAI_API_KEY in .env to run evaluation.")

    graph, _, ingestion = get_runtime(f"evaluation-{uuid.uuid4()}")
    pdf_paths = sorted(get_settings().upload_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found under {get_settings().upload_dir}. Upload PDFs in the app before evaluating.")
    for pdf_path in pdf_paths:
        ingestion.ingest(pdf_path)
    settings = get_settings()
    judge = ChatOpenAI(model=settings.router_model, temperature=0, api_key=settings.openai_api_key.get_secret_value()).with_structured_output(JudgeScores)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Score answer correctness and faithfulness to the supplied context from 0 to 1. Faithfulness means every factual claim is supported by context."),
            ("human", "Question: {question}\nExpected answer: {expected}\nActual answer: {answer}\nContext: {context}"),
        ]
    )

    expected_sources: list[set[tuple[str, int]]] = []
    actual_sources: list[set[tuple[str, int]]] = []
    expected_missing: list[bool] = []
    actual_missing: list[bool] = []
    answer_scores: list[float] = []
    faithfulness_scores: list[float] = []

    for index, example in enumerate(examples):
        question = str(example["question"])
        state = graph.invoke(
            {"question": question, "messages": [HumanMessage(content=question)], "streaming": False, "filters": {}},
            {"configurable": {"thread_id": f"evaluation-{index}"}},
        )
        actual = {(chunk.file_name, chunk.page) for chunk in state.get("chunks", [])}
        expected = {(str(source["file_name"]), int(source["page"])) for source in example.get("expected_sources", [])}
        if expected:
            expected_sources.append(expected)
            actual_sources.append(actual)
        expected_missing.append(bool(example.get("unanswerable", False)))
        actual_missing.append(state.get("route") == "rag_query" and state.get("error") == "insufficient_context")

        context = "\n\n".join(chunk.text for chunk in state.get("chunks", []))
        scores: JudgeScores = (prompt | judge).invoke(
            {"question": question, "expected": example.get("expected_answer", ""), "answer": state.get("answer", ""), "context": context}
        )
        answer_scores.append(scores.answer_score)
        faithfulness_scores.append(scores.faithfulness_score)

    print(f"Examples: {len(examples)}")
    print(f"Retrieval hit rate: {retrieval_hit_rate(expected_sources, actual_sources):.3f}")
    print(f"Answer score: {mean_score(answer_scores):.3f}")
    print(f"Faithfulness: {mean_score(faithfulness_scores):.3f}")
    print(f"Not-found accuracy: {not_found_accuracy(expected_missing, actual_missing):.3f}")


if __name__ == "__main__":
    main()
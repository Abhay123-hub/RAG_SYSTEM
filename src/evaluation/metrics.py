from collections.abc import Sequence


def retrieval_hit_rate(expected: Sequence[set[tuple[str, int]]], actual: Sequence[set[tuple[str, int]]]) -> float:
    if not expected:
        return 0.0
    hits = sum(bool(expected_sources & actual_sources) for expected_sources, actual_sources in zip(expected, actual, strict=True))
    return hits / len(expected)


def not_found_accuracy(expected: Sequence[bool], actual: Sequence[bool]) -> float:
    if not expected:
        return 0.0
    correct = sum(wanted == predicted for wanted, predicted in zip(expected, actual, strict=True))
    return correct / len(expected)


def mean_score(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0
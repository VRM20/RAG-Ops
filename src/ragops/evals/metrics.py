"""RAG quality metrics (DeepEval, LLM-judged) + deterministic retrieval metrics.

The RAG triad, plus retrieval coverage:

| metric                 | question it answers                                         | needs reference |
|------------------------|-------------------------------------------------------------|-----------------|
| faithfulness           | Is every claim in the answer supported by retrieved context? | no              |
| answer_relevancy       | Does the answer address the question (no padding/drift)?     | no              |
| contextual_precision   | Are the relevant chunks ranked above irrelevant ones?        | yes             |
| contextual_recall      | Does the retrieved context cover the reference answer?       | yes             |
| doc_recall (no LLM)    | Did every expected source document make it into context?     | source doc ids  |

Faithfulness catches hallucination (generation failure); contextual recall catches
missing evidence (retrieval failure). They are the two blocking gates in CI.
"""

from __future__ import annotations

import asyncio

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from ragops.evals.judge import RouterJudge
from ragops.evals.models import MetricScore

LLM_METRICS = ("faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall")
NEEDS_REFERENCE = {"contextual_precision", "contextual_recall"}

_FACTORIES = {
    "faithfulness": FaithfulnessMetric,
    "answer_relevancy": AnswerRelevancyMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_recall": ContextualRecallMetric,
}


def _metric(name: str, judge: RouterJudge, threshold: float):
    # Metric objects are stateful (they hold score/reason): one instance per measurement.
    return _FACTORIES[name](threshold=threshold, model=judge, include_reason=True, async_mode=True)


async def _measure(name: str, test_case: LLMTestCase, judge: RouterJudge, threshold: float) -> MetricScore:
    metric = _metric(name, judge, threshold)
    try:
        await metric.a_measure(test_case, _show_indicator=False)
        return MetricScore(score=round(float(metric.score), 4), reason=metric.reason)
    except Exception as exc:
        return MetricScore(error=f"{type(exc).__name__}: {exc}"[:300])


async def score_answer(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    expected_answer: str | None,
    judge: RouterJudge,
    thresholds: dict[str, float],
    metrics: tuple[str, ...] = LLM_METRICS,
) -> dict[str, MetricScore]:
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected_answer,
        retrieval_context=contexts or [""],
    )
    names = [m for m in metrics if expected_answer or m not in NEEDS_REFERENCE]
    scores = await asyncio.gather(*(_measure(n, test_case, judge, thresholds.get(n, 0.5)) for n in names))
    return dict(zip(names, scores, strict=True))


def doc_recall(expected_doc_ids: list[str], retrieved_doc_ids: list[str]) -> bool:
    return set(expected_doc_ids).issubset(set(retrieved_doc_ids))

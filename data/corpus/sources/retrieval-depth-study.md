# Effect of Retrieval Depth and Reranking on Grounded Answer Quality

> Fictional research report created for the RAG-Ops demo corpus. Authors: M. Okafor, L. Brandt and S. Iyer, Applied Retrieval Group. Technical report ARG-TR-2026-07.

## Abstract

We measure how retrieval strategy and the number of context passages affect the faithfulness of answers produced by a retrieval-augmented generation system. Across 1,200 questions over three document collections, hybrid retrieval with reciprocal rank fusion and a cross-encoder reranker achieved the highest recall at 5 (0.91) and the highest faithfulness (0.90), at the cost of a 410 ms median retrieval latency.

## 1. Introduction

Retrieval-augmented generation grounds a language model in retrieved passages. Retrieving too few passages misses evidence, while retrieving too many introduces distractors that the model may wrongly use. This study quantifies both effects.

## 2. Experimental Setup

### 2.1 Dataset

The evaluation set contains 1,200 questions with reference answers written by domain experts: 400 over legal contracts, 400 over technical product manuals and 400 over human-resources policies.

### 2.2 Retrieval Configurations

Four configurations were compared: dense-only retrieval, BM25-only retrieval, hybrid retrieval fusing both rankings with reciprocal rank fusion (k = 60), and the same hybrid retrieval followed by a cross-encoder reranker. Unless stated otherwise, the generator received the top 5 passages.

### 2.3 Metrics

Recall at 5 is the fraction of questions for which all required evidence appears in the top 5 passages. Faithfulness is the fraction of answer claims supported by the retrieved passages, scored by an LLM judge. Latency is the median (p50) retrieval time per query.

## 3. Results

### 3.1 Retrieval Configuration

| Configuration | Recall at 5 | Faithfulness | Median latency |
| --- | --- | --- | --- |
| Dense only | 0.71 | 0.82 | 140 ms |
| BM25 only | 0.64 | 0.79 | 35 ms |
| Hybrid (RRF) | 0.83 | 0.86 | 165 ms |
| Hybrid + reranker | 0.91 | 0.90 | 410 ms |

BM25 performed best on the legal contracts, where exact clause identifiers matter, while dense retrieval performed best on the human-resources policies, where questions paraphrase the policy text.

### 3.2 Number of Context Passages

With hybrid retrieval and reranking, faithfulness peaked at 5 passages (0.90). Increasing the context to 15 passages lowered faithfulness to 0.84, because the generator began citing loosely related passages. Reducing the context to 2 passages lowered recall at 5 to 0.74.

## 4. Discussion

The reranker added 8 points of recall at 5 over plain hybrid retrieval, for an extra 245 ms of median latency. For interactive assistants this trade-off is acceptable; for latency-critical autocomplete use cases, plain hybrid retrieval is the better default.

## 5. Limitations

All questions were in English, and faithfulness was scored by a single judge model. Results may differ for multilingual collections and for other judge models.

## 6. Conclusion

We recommend hybrid retrieval with reciprocal rank fusion, a cross-encoder reranker and 5 context passages as the default configuration, and evaluating any change to retrieval depth against a golden dataset before deployment.

## References

1. Cormack, G., Clarke, C. and Buettcher, S. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR 2009.
2. Robertson, S. and Zaragoza, H. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 2009.

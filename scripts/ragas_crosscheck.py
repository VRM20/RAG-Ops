"""Cross-check a DeepEval run with RAGAS (judge-agreement analysis).

Why: an LLM-judged gate is only as trustworthy as its judge. Re-scoring the *same*
answers and contexts with an independent framework (different prompts, different
claim-decomposition logic) shows whether the gate's numbers are stable or artefacts
of one framework's prompting.

Runs in its own virtualenv on purpose: RAGAS pins langchain/instructor versions that
conflict with the production stack, so it never shares an environment with the API.
The contract between the two is just the persisted run JSON.

    python3.13 -m venv ~/.venvs/ragas
    ~/.venvs/ragas/bin/pip install -r requirements-ragas.txt
    ~/.venvs/ragas/bin/python scripts/ragas_crosscheck.py --run latest --max-cases 15

Output: per-case DeepEval vs RAGAS scores for faithfulness and context recall, the
mean absolute difference, Pearson / Spearman correlation and pass/fail agreement at
the gate threshold. Written to <run>.ragas.json next to the input run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {"faithfulness": "faithfulness", "contextual_recall": "context_recall"}


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass


def _find_run(arg: str) -> Path:
    if arg != "latest":
        return Path(arg)
    candidates = list((ROOT / "data" / "state" / "eval_runs").glob("*.json")) + list((ROOT / "evals" / "reports").glob("run.json"))
    if not candidates:
        raise SystemExit("No eval runs found. Run `ragops eval` first.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _vertex_kwargs() -> dict:
    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "secrets/gcp-vertex-key.json")
    key_path = Path(key) if Path(key).is_absolute() else ROOT / key
    return {
        "vertex_project": os.environ.get("GCP_PROJECT_ID"),
        "vertex_location": os.environ.get("VERTEX_LOCATION", "global"),
        "vertex_credentials": key_path.read_text(encoding="utf-8"),
    }


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _agreement(a: list[float], b: list[float], threshold: float) -> dict:
    out: dict = {"n": len(a)}
    if len(a) < 2:
        return out
    out["mean_abs_diff"] = round(statistics.fmean(abs(x - y) for x, y in zip(a, b, strict=True)), 4)
    try:
        out["pearson"] = round(statistics.correlation(a, b), 4)
        out["spearman"] = round(statistics.correlation(_ranks(a), _ranks(b)), 4)
    except statistics.StatisticsError:  # constant series (e.g. every score is 1.0)
        out["pearson"] = out["spearman"] = None
    out["pass_fail_agreement"] = round(statistics.fmean(float((x >= threshold) == (y >= threshold)) for x, y in zip(a, b, strict=True)), 4)
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="latest", help="path to a run JSON, or 'latest'")
    parser.add_argument("--max-cases", type=int, default=15)
    parser.add_argument("--model", default=os.environ.get("RAGAS_JUDGE_MODEL", "vertex_ai/gemini-3.8-flash"))
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    _load_env()

    import instructor
    import litellm
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall, Faithfulness

    run_path = _find_run(args.run)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    cases = [c for c in run["cases"] if not c.get("error") and c.get("metrics")][: args.max_cases]
    print(f"RAGAS cross-check of {run_path.name}: {len(cases)} cases, judge {args.model}")

    extra = _vertex_kwargs() if args.model.startswith("vertex_ai/") else {}
    llm = llm_factory(
        args.model,
        provider="litellm",
        client=instructor.from_litellm(litellm.acompletion),
        adapter="litellm",
        max_tokens=8192,
        **extra,
    )
    faithfulness, recall = Faithfulness(llm=llm), ContextRecall(llm=llm)
    sem = asyncio.Semaphore(args.concurrency)

    async def score(case: dict) -> dict:
        contexts = [c["text"] for c in case["contexts"]] or [""]
        async with sem:
            try:
                f = await faithfulness.ascore(user_input=case["question"], response=case["answer"], retrieved_contexts=contexts)
                r = await recall.ascore(user_input=case["question"], retrieved_contexts=contexts, reference=case["expected_answer"])
                ragas = {"faithfulness": float(f.value), "context_recall": float(r.value)}
            except Exception as exc:
                ragas = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        deepeval = {k: (case["metrics"].get(k) or {}).get("score") for k in PAIRS}
        return {"case_id": case["case_id"], "deepeval": deepeval, "ragas": ragas}

    rows = await asyncio.gather(*(score(c) for c in cases))
    summary = {}
    for de_name, rg_name in PAIRS.items():
        pairs = [(r["deepeval"][de_name], r["ragas"][rg_name]) for r in rows if r["deepeval"].get(de_name) is not None and rg_name in r["ragas"]]
        summary[de_name] = _agreement([p[0] for p in pairs], [p[1] for p in pairs], args.threshold)

    out = {"run": run_path.name, "judge": args.model, "summary": summary, "cases": rows}
    out_path = run_path.with_suffix(".ragas.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{'case':<34}{'faith DE':>10}{'faith RG':>10}{'recall DE':>11}{'recall RG':>11}")
    for r in rows:
        de, rg = r["deepeval"], r["ragas"]
        fmt = lambda v: "   err" if v is None else f"{v:>6.2f}"  # noqa: E731
        print(f"{r['case_id']:<34}{fmt(de['faithfulness']):>10}{fmt(rg.get('faithfulness')):>10}{fmt(de['contextual_recall']):>11}{fmt(rg.get('context_recall')):>11}")
    print("\nAgreement:", json.dumps(summary, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

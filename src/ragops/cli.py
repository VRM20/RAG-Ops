"""``ragops`` command-line interface.

    ragops doctor                     verify every credential and service, one by one
    ragops build-pdfs                 render the PDF documents of the demo corpus
    ragops ingest                     ingest the demo corpus (data/corpus/manifest.yaml)
    ragops ingest --file f.pdf --acl hr,leadership
    ragops ingest --dir ./my-pdfs --acl all-employees
    ragops ask "question" --user priya.sharma
    ragops synth                      generate the synthetic golden dataset with Gemini
    ragops eval [--max-cases N] [--save-baseline] [--report out.md] [--fail-on-gate]
    (RAGAS cross-check: scripts/ragas_crosscheck.py, isolated venv)
    ragops create-guardrail           provision the AWS Bedrock Guardrail
    ragops serve                      run the API (uvicorn)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from ragops.config import ROOT_DIR, get_settings

OK, FAIL, SKIP = "\033[32m✔\033[0m", "\033[31m✘\033[0m", "\033[33m•\033[0m"


# ---- doctor ----------------------------------------------------------------------------
async def _doctor() -> int:
    from ragops.config import get_models_config, get_rag_config

    s = get_settings()
    failures = 0

    def report(ok: bool | None, name: str, detail: str = "") -> None:
        nonlocal failures
        mark = SKIP if ok is None else OK if ok else FAIL
        failures += ok is False
        print(f"  {mark} {name:<34} {detail}")

    print(f"\nRAG-Ops doctor  (env={s.ragops_env}, offline={s.ragops_offline})\n")
    models, rag = get_models_config(), get_rag_config()

    # GCP
    if s.gcp_key_path is None or not s.gcp_project_id:
        report(None, "GCP service account", "not configured (GCP_PROJECT_ID / GOOGLE_APPLICATION_CREDENTIALS)")
    else:
        try:
            from ragops.providers.gcp import get_credentials, init_vertex

            creds = get_credentials()
            init_vertex()
            report(True, "GCP service account", f"{creds.service_account_email}")
        except Exception as exc:
            report(False, "GCP service account", str(exc)[:160])
        # Probe only the Vertex models a route actually uses (parked ones are skipped).
        used = {name for route in models.routes.values() for name in [route.primary, *route.fallbacks]}
        for key, spec in models.models.items():
            if spec.provider == "vertex" and key in used:
                await _probe_llm(report, f"Vertex AI {key}", spec)
        await _probe_embedding(report, rag)

    # AWS
    if not s.has_aws:
        report(None, "AWS credentials", "not configured (AWS_ACCESS_KEY_ID / AWS_PROFILE / IAM role)")
    else:
        try:
            from ragops.providers.aws import get_session

            ident = get_session().client("sts").get_caller_identity()
            report(True, "AWS credentials", f"{ident['Arn']} ({s.aws_region})")
        except Exception as exc:
            report(False, "AWS credentials", str(exc)[:160])
        await _probe_llm(report, "Bedrock Claude (haiku)", models.models["claude-haiku"])
        await _probe_llm(report, "Bedrock Claude (sonnet)", models.models["claude-sonnet"])
        if s.bedrock_guardrail_id:
            try:
                from ragops.providers.guardrails import Guardrails

                v = await Guardrails().check_input("My PAN is ABCDE1234F, how do I claim travel expenses?")
                report(v.provider == "bedrock", "Bedrock Guardrail", f"{s.bedrock_guardrail_id} v{s.bedrock_guardrail_version}: {v.action} {v.reasons}")
            except Exception as exc:
                report(False, "Bedrock Guardrail", str(exc)[:160])
        else:
            report(None, "Bedrock Guardrail", "BEDROCK_GUARDRAIL_ID not set: run `ragops create-guardrail` (local guard in use)")

    # Cohere
    if s.ragops_offline:
        report(None, "Cohere Rerank", "offline mode: local lexical reranker in use")
    elif rag.retrieval.rerank.enabled and rag.retrieval.rerank.model.startswith("cohere/"):
        if not s.cohere_api_key:
            report(False, "Cohere Rerank", "COHERE_API_KEY not set")
        else:
            try:
                import litellm

                r = await litellm.arerank(
                    model=rag.retrieval.rerank.model, query="annual leave", documents=["Employees earn 24 days of leave.", "USB is disabled."],
                    top_n=1, api_key=s.cohere_api_key.get_secret_value(),
                )
                report(True, "Cohere Rerank", f"{rag.retrieval.rerank.model} top score {r.results[0]['relevance_score']:.3f}")
            except Exception as exc:
                report(False, "Cohere Rerank", str(exc)[:160])

    # Infra
    try:
        from ragops.retrieval.store import get_qdrant_client, get_store

        await get_qdrant_client().get_collections()
        report(True, "Qdrant", f"{'server ' + s.qdrant_url if s.qdrant_url else 'embedded ' + str(s.qdrant_path)} | index {get_store().collection}")
    except Exception as exc:
        report(False, "Qdrant", str(exc)[:160])
    if s.task_queue == "celery":
        try:
            import redis

            redis.Redis.from_url(s.redis_url).ping()
            report(True, "Redis", s.redis_url)
        except Exception as exc:
            report(False, "Redis", str(exc)[:160])
    else:
        report(None, "Redis", "TASK_QUEUE=inline (not needed)")
    report(None if not s.otel_exporter_otlp_traces_endpoint else True, "Tracing (OTLP)", s.otel_exporter_otlp_traces_endpoint or "disabled")
    print(f"\n{'All checks passed.' if not failures else f'{failures} check(s) failed.'}\n")
    return 1 if failures else 0


async def _probe_llm(report, name: str, spec) -> None:
    import litellm

    from ragops.providers.router import LLMRouter

    try:
        params = {"model": spec.litellm_model, **spec.params, **LLMRouter._credentials(spec)}
        resp = await litellm.acompletion(messages=[{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=512, **params)
        report(True, name, f"{spec.litellm_model} -> {(resp.choices[0].message.content or '').strip()[:30]!r}")
    except Exception as exc:
        report(False, name, f"{spec.litellm_model}: {str(exc)[:150]}")


async def _probe_embedding(report, rag) -> None:
    from ragops.providers.embeddings import Embedder

    try:
        vec = await Embedder(rag.embedding).embed_query("privileged leave carry forward")
        report(True, "Embeddings", f"{rag.embedding.model} ({len(vec)} dims)")
    except Exception as exc:
        report(False, "Embeddings", f"{rag.embedding.model}: {str(exc)[:150]}")


# ---- commands --------------------------------------------------------------------------
def cmd_build_pdfs(_: argparse.Namespace) -> int:
    from ragops.ingestion.pipeline import corpus_file, load_manifest

    for doc in load_manifest():
        if doc.format == "pdf":
            print("  rendered", corpus_file(doc).relative_to(ROOT_DIR))
    return 0


async def cmd_ingest(args: argparse.Namespace) -> int:
    from ragops.ingestion.pipeline import IngestionService

    svc = IngestionService()
    if args.file or args.dir:
        from ragops.ingestion.parsers import SUPPORTED_SUFFIXES

        if args.file:
            paths = [Path(args.file)]
        else:
            paths = sorted(p for p in Path(args.dir).rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
            print(f"Found {len(paths)} supported file(s) under {args.dir}")
        records = []
        for path in paths:
            title = args.title if args.file else None
            # In --dir mode the relative path keeps ids unique across sub-folders.
            doc_id = None if args.file else str(path.relative_to(args.dir).with_suffix(""))
            rec = svc.register(filename=path.name, data=path.read_bytes(), acl_groups=args.acl.split(","), title=title, doc_id=doc_id)
            records.append(await svc.process(rec.doc_id))
    else:
        records = await svc.ingest_corpus()
    print(f"\nIndex: {svc.index_name}")
    for r in records:
        mark = OK if r.status == "indexed" else FAIL
        print(f"  {mark} {r.doc_id:<40} {r.status:<8} chunks={r.num_chunks:<3} parser={r.parser} {r.error or ''}")
    return 0 if all(r.status == "indexed" for r in records) else 1


async def cmd_ask(args: argparse.Namespace) -> int:
    from ragops.generation.pipeline import RAGPipeline
    from ragops.security.acl import resolve_user

    resp = await RAGPipeline().answer(args.question, resolve_user(args.user))
    print(f"\n{resp.answer}\n")
    for i, c in enumerate(resp.contexts, start=1):
        where = " | ".join(x for x in (c.chunk.doc_title, c.chunk.section or "Overview", c.chunk.page_label) if x)
        print(f"  [{i}] {where}  (rerank={c.rerank_score}, dense#{c.dense_rank}, bm25#{c.sparse_rank})")
    print(f"\n  model={resp.model}  timings={json.dumps(resp.timings_ms)}  guardrails={[g.action for g in resp.guardrails]}")
    return 0


async def cmd_eval(args: argparse.Namespace) -> int:
    from ragops.evals.report import render_markdown
    from ragops.evals.runner import load_dataset, run_evaluation
    from ragops.evals.store import BASELINE_PATH, save_run

    cases = load_dataset(dataset=args.dataset, max_cases=args.max_cases)
    print(f"Evaluating {len(cases)} cases ...")
    run = await run_evaluation(
        label=args.label, cases=cases, on_progress=lambda d, t: print(f"\r  {d}/{t}", end="", flush=True)
    )
    print()
    md = render_markdown(run)
    print(md)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(md, encoding="utf-8")
    if args.json:
        save_run(run, Path(args.json))
    if args.save_baseline:
        if run.status != "completed":
            print("Run did not complete; baseline not saved.")
            return 1
        save_run(run, BASELINE_PATH)
        print(f"Baseline saved to {BASELINE_PATH.relative_to(ROOT_DIR)}")
    if run.status != "completed":
        return 2
    return 1 if (args.fail_on_gate and not run.gate_passed) else 0


async def cmd_synth(args: argparse.Namespace) -> int:
    from ragops.evals.synthesizer import generate_synthetic_dataset

    cases = await generate_synthetic_dataset(per_section=args.per_section, critic=not args.no_critic)
    print(f"Generated {len(cases)} synthetic cases -> evals/datasets/golden_synthetic.jsonl")
    return 0




def cmd_create_guardrail(args: argparse.Namespace) -> int:
    from ragops.providers.guardrail_setup import create_guardrail

    gid, version = create_guardrail(args.name)
    print(f"\nGuardrail ready. Add to .env:\n\n  BEDROCK_GUARDRAIL_ID={gid}\n  BEDROCK_GUARDRAIL_VERSION={version}\n")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("ragops.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragops", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="verify credentials and services")
    sub.add_parser("build-pdfs", help="render corpus PDFs")

    p = sub.add_parser("ingest", help="ingest the demo corpus or a single file")
    p.add_argument("--file", help="ingest one .pdf/.md/.txt file")
    p.add_argument("--dir", help="ingest every supported file under this folder (recursive)")
    p.add_argument("--acl", default="public", help="comma-separated ACL groups for --file/--dir")
    p.add_argument("--title")

    p = sub.add_parser("ask", help="ask one question from the terminal")
    p.add_argument("question")
    p.add_argument("--user", default="eval.bot")

    p = sub.add_parser("synth", help="generate the synthetic golden dataset (Gemini)")
    p.add_argument("--per-section", type=int, default=2)
    p.add_argument("--no-critic", action="store_true")

    p = sub.add_parser("eval", help="run the evaluation suite")
    p.add_argument("--label", default="cli")
    p.add_argument("--dataset", choices=["all", "curated", "synthetic"], default="all")
    p.add_argument("--max-cases", type=int)
    p.add_argument("--report", help="write the markdown report to this path")
    p.add_argument("--json", help="write the full run JSON to this path")
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--fail-on-gate", action="store_true")


    p = sub.add_parser("create-guardrail", help="provision the AWS Bedrock Guardrail")
    p.add_argument("--name", default="ragops-enterprise-rag")

    p = sub.add_parser("serve", help="run the API server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "LiteLLM", "litellm", "urllib3", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sync = {"build-pdfs": cmd_build_pdfs, "create-guardrail": cmd_create_guardrail, "serve": cmd_serve}
    async_cmds = {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval, "synth": cmd_synth}
    if args.command == "doctor":
        return asyncio.run(_doctor())
    if args.command in sync:
        return sync[args.command](args)
    return asyncio.run(async_cmds[args.command](args))


if __name__ == "__main__":
    sys.exit(main())

"""CLI for Formulagate gate / RAG / eval / bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from formulagate.audit import build_rag_audit
from formulagate.bench import benchmark_rerankers
from formulagate.corpus import load_corpus
from formulagate.gate import run_gate
from formulagate.metrics import evaluate_golden_cases, load_golden_cases


def _apply_calibration(path: Path | None) -> None:
    """Install a fitted calibration artifact before any gate decision runs."""

    if path is None:
        return
    from formulagate.calibration import CalibrationError, load_calibration

    try:
        load_calibration(path)
    except CalibrationError as exc:
        raise SystemExit(f"calibration error: {exc}") from exc


def _apply_multi_calibration(path: Path | None) -> None:
    """Install a fused (similarity + physics) model, if one was supplied."""

    if path is None:
        return
    from formulagate.calibration import (
        CalibrationError,
        load_multi_calibration,
        set_multi_calibration,
    )

    try:
        set_multi_calibration(load_multi_calibration(path))
    except CalibrationError as exc:
        raise SystemExit(f"fused calibration error: {exc}") from exc


def _print_gate(evaluation: Any, as_json: bool) -> None:
    if as_json:
        payload = {
            "ok": evaluation.ok,
            "detail": evaluation.detail,
            "confidence": evaluation.confidence,
            "threshold": evaluation.threshold,
            "confidence_model": evaluation.model,
            "physics": evaluation.physics,
            "domain": evaluation.result.domain,
            "entries": [
                {
                    "record_id": e.record_id,
                    "score": e.score,
                    "math_relevance": e.math_relevance,
                    "formula_excerpt": e.formula_excerpt,
                    "scientific_domain": e.scientific_domain,
                    "meta": e.meta,
                }
                for e in evaluation.result.entries
            ],
            "formula": "score = overlap + 2*rel",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print("ok:", evaluation.ok)
    print("detail:", evaluation.detail)
    if evaluation.confidence is not None:
        print(
            f"confidence: {evaluation.confidence:.3f} "
            f"(threshold {evaluation.threshold:.2f}, model {evaluation.model})"
        )
    if evaluation.physics:
        print(
            "physics: dimensions={dimension_ok:+.0f} equivalence={equivalence:+.0f} "
            "symbols={symbol_overlap:.2f}".format(**evaluation.physics)
        )
    print("domain:", evaluation.result.domain)
    if evaluation.result.entries:
        top = evaluation.result.entries[0]
        print(
            "top:",
            top.record_id,
            "score=",
            top.score,
            "rel=",
            top.math_relevance,
        )
        print("formula:", top.formula_excerpt)


def _cmd_verify(args: argparse.Namespace) -> int:
    """Check one formula on its own: is it dimensionally possible, and does it
    agree with a reference? No corpus, no embeddings, no model — just algebra."""

    from formulagate.dimensions import check_dimensions
    from formulagate.equivalence import check_equivalence
    from formulagate.formula_extract import canonicalize

    formula = canonicalize(args.formula)
    dimensions = check_dimensions(formula)
    payload: dict[str, Any] = {
        "formula": formula.latex,
        "parsed": formula.is_usable,
        "parse_error": formula.parse_error,
        "symbols": list(formula.symbols),
        "structure_hash": formula.structure_hash,
        "dimensions": dimensions.to_dict(),
    }
    if args.against:
        payload["equivalence"] = check_equivalence(
            formula, canonicalize(args.against)
        ).to_dict()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("formula:", payload["formula"])
        if not formula.is_usable:
            print("parse error:", formula.parse_error)
        else:
            print("symbols:", ", ".join(formula.symbols) or "-")
            print(f"dimensions: {dimensions.status} — {dimensions.detail}")
            if args.against:
                eq = payload["equivalence"]
                print(f"equivalence: {eq['status']} via {eq['method']} — {eq['detail']}")

    return 1 if dimensions.is_reject else 0


def _cmd_gate(args: argparse.Namespace) -> int:
    _apply_calibration(args.calibration)
    _apply_multi_calibration(args.calibration_multi)
    evaluation = run_gate(
        brief=args.brief,
        candidate_text=args.draft,
        corpus_path=args.corpus,
        top_k=args.top_k,
        use_physics=not args.no_physics,
    )
    _print_gate(evaluation, args.json)
    return 0 if evaluation.ok else 1


def _make_embedder(name: str) -> Any:
    if name == "hash":
        from formulagate.dense import HashEmbedder

        return HashEmbedder(dim=64)
    if name == "minilm":
        from formulagate.dense import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder()
    raise SystemExit(f"unknown embedder: {name}")


def _make_reranker(name: str) -> Any | None:
    if name == "none":
        return None
    if name == "formulagate":
        from formulagate.rerank import FormulagateReranker

        return FormulagateReranker()
    if name == "cross-encoder":
        from formulagate.rerank import CrossEncoderReranker

        return CrossEncoderReranker()
    raise SystemExit(f"unknown reranker: {name}")


def _make_dense_retriever(corpus: list[dict[str, Any]], embedder_name: str) -> Any:
    from formulagate.dense import DenseHybridRetriever

    return DenseHybridRetriever(corpus, embedder=_make_embedder(embedder_name))


def _cmd_rag(args: argparse.Namespace) -> int:
    from formulagate.rag import run_scientific_rag

    _apply_calibration(args.calibration)
    _apply_multi_calibration(args.calibration_multi)
    rows = load_corpus(args.corpus)
    retriever = _make_dense_retriever(rows, args.embedder)
    reranker = _make_reranker(args.reranker)
    decision = run_scientific_rag(
        brief=args.brief,
        draft=args.draft,
        retriever=retriever,
        retrieve_k=args.top_k,
        retrieve_pool=args.pool,
        reranker=reranker,
    )
    audit = build_rag_audit(
        brief=args.brief,
        draft=args.draft,
        decision=decision,
        corpus_path=str(Path(args.corpus).resolve()),
        embedder=args.embedder,
        reranker=args.reranker,
    )
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.json or args.audit:
        text = json.dumps(audit, ensure_ascii=False, indent=2)
        try:
            print(text)
        except UnicodeEncodeError:
            print(json.dumps(audit, ensure_ascii=True, indent=2))
    else:
        print("action:", decision.action)
        print("ok:", decision.ok)
        print("detail:", decision.gate.detail)
        if decision.gate.confidence is not None:
            print(f"confidence: {decision.gate.confidence:.3f}")
        print("domain:", decision.gate.result.domain)
        print("retrieved:", decision.retrieved_ids)
        if decision.gate.result.entries:
            top = decision.gate.result.entries[0]
            print("top:", top.record_id, "score=", top.score)
            print("formula:", top.formula_excerpt)
    return 0 if decision.ok else 1


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Replay the golden set through the gate and fit (a, b, threshold)."""

    from formulagate.calibration import (
        CalibrationError,
        collect_calibration_samples,
        evaluate_calibration,
        fit_calibration,
        save_calibration,
        set_calibration,
    )

    cases = load_golden_cases(args.golden)
    corpus = load_corpus(args.corpus)
    if args.corpus_limit:
        corpus = corpus[: args.corpus_limit]
    reranker = _make_reranker(args.reranker)
    scores, labels = collect_calibration_samples(
        cases,
        corpus=corpus,
        make_retriever=lambda rows: _make_dense_retriever(rows, args.embedder),
        retrieve_k=args.top_k,
        reranker=reranker,
        limit=args.limit,
    )
    try:
        params = fit_calibration(
            scores, labels, l2_penalty=args.l2, beta=args.beta, objective=args.objective
        )
        report = evaluate_calibration(scores, labels, params, optimize_threshold=False)
    except CalibrationError as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 1

    set_calibration(params)
    save_calibration(args.out, params, report)
    payload = {"artifact": str(args.out), **report.to_dict()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.expected_calibration_error <= args.max_ece else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    _apply_calibration(args.calibration)
    cases = load_golden_cases(args.golden)
    corpus = load_corpus(args.corpus)
    reranker = _make_reranker(args.reranker)
    report = evaluate_golden_cases(
        cases,
        corpus=corpus,
        make_retriever=lambda rows: _make_dense_retriever(rows, args.embedder),
        retrieve_k=args.top_k,
        reranker=reranker,
    )
    payload = {
        "n": report.n,
        "accuracy": report.accuracy,
        "generate_precision": report.generate_precision,
        "abstain_recall": report.abstain_recall,
        "top_id_hits": report.top_id_hits,
        "top_id_checked": report.top_id_checked,
        "embedder": args.embedder,
        "reranker": args.reranker,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    ok = (
        report.n >= args.min_n
        and report.accuracy >= args.min_accuracy
        and report.abstain_recall >= args.min_abstain_recall
    )
    return 0 if ok else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    cases = load_golden_cases(args.golden)
    corpus = load_corpus(args.corpus)
    result = benchmark_rerankers(
        cases,
        corpus=corpus,
        make_retriever=lambda rows: _make_dense_retriever(rows, args.embedder),
        retrieve_k=args.top_k,
    )
    payload = {
        "embedder": args.embedder,
        "formulagate": {
            "accuracy": result.formulagate.accuracy,
            "abstain_recall": result.formulagate.abstain_recall,
            "top_id": f"{result.formulagate.top_id_hits}/{result.formulagate.top_id_checked}",
        },
        "cross_encoder": {
            "accuracy": result.cross_encoder.accuracy,
            "abstain_recall": result.cross_encoder.abstain_recall,
            "top_id": f"{result.cross_encoder.top_id_hits}/{result.cross_encoder.top_id_checked}",
        },
        "accuracy_delta": result.accuracy_delta,
        "top_id_delta": result.top_id_delta,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


# ─── Admin commands ────────────────────────────────────────────────────────────


def _admin_key_create(args: argparse.Namespace) -> int:
    """Create an API key for a new or existing user."""
    from formulagate.auth import generate_api_key, hash_api_key
    from formulagate.database import get_database

    db = get_database()
    if not db.enabled:
        print("Error: PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)", file=sys.stderr)
        return 1

    api_key = generate_api_key()
    hashed = hash_api_key(api_key)
    prefix = api_key[:15]

    user = db.create_user(
        email=args.email,
        api_key_hash=hashed,
        api_key_prefix=prefix,
        plan=args.plan,
    )
    if user is None:
        print(f"Error: user with email {args.email} already exists", file=sys.stderr)
        return 1

    print(json.dumps({
        "api_key": api_key,
        "api_key_prefix": prefix,
        "user_id": user["id"],
        "email": user["email"],
        "plan": user["plan"],
    }, indent=2))
    return 0


def _admin_key_revoke(args: argparse.Namespace) -> int:
    """Revoke (delete) a user by email or user id."""
    from formulagate.database import get_database

    db = get_database()
    if not db.enabled:
        print("Error: PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)", file=sys.stderr)
        return 1

    if args.user_id:
        user = db.get_user_by_id(args.user_id)
    else:
        user = db.get_user_by_email(args.email)

    if user is None:
        print("Error: user not found", file=sys.stderr)
        return 1

    ok = db.revoke_user(user["id"])
    if not ok:
        print("Error: failed to revoke user", file=sys.stderr)
        return 1

    print(json.dumps({"revoked": user["id"], "email": user["email"]}, indent=2))
    return 0


def _admin_key_list(args: argparse.Namespace) -> int:
    """List registered users."""
    from formulagate.database import get_database

    db = get_database()
    if not db.enabled:
        print("Error: PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)", file=sys.stderr)
        return 1

    users = db.list_users(limit=args.limit, offset=args.offset)
    print(json.dumps({"users": users, "count": len(users)}, indent=2))
    return 0


def _admin_plan_set(args: argparse.Namespace) -> int:
    """Set a user's plan."""
    from formulagate.database import get_database
    from formulagate.plans import PLANS

    if args.plan not in PLANS:
        print(f"Error: unknown plan {args.plan!r}", file=sys.stderr)
        return 1

    db = get_database()
    if not db.enabled:
        print("Error: PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)", file=sys.stderr)
        return 1

    if args.user_id:
        user = db.get_user_by_id(args.user_id)
    else:
        user = db.get_user_by_email(args.email)

    if user is None:
        print("Error: user not found", file=sys.stderr)
        return 1

    ok = db.update_user_plan(user["id"], args.plan)
    if not ok:
        print("Error: failed to update plan", file=sys.stderr)
        return 1

    print(json.dumps({"user_id": user["id"], "email": user["email"], "plan": args.plan}, indent=2))
    return 0


def _admin_plan_info(args: argparse.Namespace) -> int:
    """Show a user's plan and usage."""
    from formulagate.database import get_database
    from formulagate.plans import PLANS

    db = get_database()
    if not db.enabled:
        print("Error: PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)", file=sys.stderr)
        return 1

    if args.user_id:
        user = db.get_user_by_id(args.user_id)
    else:
        user = db.get_user_by_email(args.email)

    if user is None:
        print("Error: user not found", file=sys.stderr)
        return 1

    usage = db.get_monthly_usage(user["id"])
    plan = PLANS.get(user["plan"], PLANS["free"])
    ok, used, limit = db.check_quota(user["id"], user["plan"])

    print(json.dumps({
        "user_id": user["id"],
        "email": user["email"],
        "plan": user["plan"],
        "plan_name": plan.name,
        "monthly_quota": plan.monthly_quota,
        "usage_this_month": usage,
        "total_used": used,
        "remaining": limit - used if limit > 0 else "unlimited",
        "subscription": db.get_active_subscription(user["id"]),
        "created_at": user["created_at"],
    }, indent=2, default=str))
    return 0


def _admin_stats(args: argparse.Namespace) -> int:
    """Show aggregate usage statistics."""
    from formulagate.database import get_database

    db = get_database()
    if not db.enabled:
        print("Error: PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)", file=sys.stderr)
        return 1

    stats = db.get_usage_stats()
    stats["db_metrics"] = db.get_stats()
    print(json.dumps(stats, indent=2))
    return 0


def _add_gate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--brief", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--calibration",
        type=Path,
        help="Fitted calibration artifact (see `formulagate calibrate`)",
    )
    p.add_argument(
        "--calibration-multi",
        type=Path,
        help="Fused similarity+physics model (see `scripts/bench_real.py --multi-out`)",
    )
    p.add_argument(
        "--no-physics",
        action="store_true",
        help="Disable the dimensional-analysis veto and physics features",
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatible: bare flags ⇒ gate subcommand.
    if not argv or argv[0].startswith("-"):
        argv = ["gate", *argv]

    parser = argparse.ArgumentParser(
        prog="formulagate",
        description="Formula IR gate + scientific RAG CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    verify_p = sub.add_parser(
        "verify",
        help="Dimensional analysis + equivalence for a single formula (no corpus)",
    )
    verify_p.add_argument("--formula", required=True, help="LaTeX, e.g. 'E = m c^2'")
    verify_p.add_argument("--against", help="Reference formula to compare with")
    verify_p.add_argument("--json", action="store_true")
    verify_p.set_defaults(func=_cmd_verify)

    gate_p = sub.add_parser("gate", help="Classic corpus-wide accept/reject gate")
    _add_gate_args(gate_p)
    gate_p.set_defaults(func=_cmd_gate)

    rag_p = sub.add_parser("rag", help="Dense retrieve → rerank → Formulagate judge")
    _add_gate_args(rag_p)
    rag_p.add_argument(
        "--embedder",
        choices=("hash", "minilm"),
        default="hash",
        help="hash=CI stub; minilm=SentenceTransformer",
    )
    rag_p.add_argument(
        "--reranker",
        choices=("formulagate", "cross-encoder", "none"),
        default="formulagate",
    )
    rag_p.add_argument("--pool", type=int, default=20, help="Retrieve pool before rerank")
    rag_p.add_argument("--audit", type=Path, help="Write JSON audit trail to path")
    rag_p.set_defaults(func=_cmd_rag)

    eval_p = sub.add_parser("eval", help="Run golden-set metrics (CI floor)")
    eval_p.add_argument("--golden", type=Path, required=True)
    eval_p.add_argument("--corpus", type=Path, required=True)
    eval_p.add_argument("--embedder", choices=("hash", "minilm"), default="hash")
    eval_p.add_argument(
        "--reranker",
        choices=("formulagate", "cross-encoder", "none"),
        default="formulagate",
    )
    eval_p.add_argument("--top-k", type=int, default=3)
    eval_p.add_argument("--min-n", type=int, default=50)
    eval_p.add_argument("--min-accuracy", type=float, default=0.75)
    eval_p.add_argument("--min-abstain-recall", type=float, default=1.0)
    eval_p.add_argument("--calibration", type=Path)
    eval_p.set_defaults(func=_cmd_eval)

    cal_p = sub.add_parser(
        "calibrate",
        help="Fit confidence calibration (Platt sigmoid + threshold) on golden cases",
    )
    cal_p.add_argument("--golden", type=Path, required=True)
    cal_p.add_argument("--corpus", type=Path, required=True)
    cal_p.add_argument("--out", type=Path, required=True, help="Artifact path")
    cal_p.add_argument("--embedder", choices=("hash", "minilm"), default="hash")
    cal_p.add_argument(
        "--reranker",
        choices=("formulagate", "cross-encoder", "none"),
        default="formulagate",
    )
    cal_p.add_argument("--top-k", type=int, default=3)
    cal_p.add_argument("--limit", type=int, help="Cap the number of golden cases")
    cal_p.add_argument(
        "--corpus-limit",
        type=int,
        help="Use only the first N corpus records (fast fits on huge corpora)",
    )
    cal_p.add_argument("--l2", type=float, default=1e-3, help="Slope shrinkage")
    cal_p.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="F-beta for threshold search (<1 favours precision)",
    )
    cal_p.add_argument(
        "--objective",
        choices=("balanced_accuracy", "fbeta"),
        default="balanced_accuracy",
        help="Threshold criterion; fbeta ignores true negatives and drifts to accept-all",
    )
    cal_p.add_argument(
        "--max-ece",
        type=float,
        default=1.0,
        help="CI floor: exit 1 when calibration error exceeds this",
    )
    cal_p.set_defaults(func=_cmd_calibrate)

    bench_p = sub.add_parser(
        "bench",
        help="Compare Formulagate vs CrossEncoder rerank on golden set",
    )
    bench_p.add_argument("--golden", type=Path, required=True)
    bench_p.add_argument("--corpus", type=Path, required=True)
    bench_p.add_argument("--embedder", choices=("hash", "minilm"), default="hash")
    bench_p.add_argument("--top-k", type=int, default=3)
    bench_p.set_defaults(func=_cmd_bench)

    # ── Admin subcommands ─────────────────────────────────────────────────
    admin_p = sub.add_parser("admin", help="Administrative commands (users, plans, stats)")
    admin_sub = admin_p.add_subparsers(dest="admin_command", required=True)

    # admin key
    key_p = admin_sub.add_parser("key", help="Manage API keys")
    key_sub = key_p.add_subparsers(dest="key_command", required=True)

    key_create = key_sub.add_parser("create", help="Create a new API key")
    key_create.add_argument("--email", required=True, help="User email address")
    key_create.add_argument("--plan", default="free", help="Plan to assign (default: free)")
    key_create.set_defaults(func=_admin_key_create)

    key_revoke = key_sub.add_parser("revoke", help="Revoke (delete) a user's API key")
    key_revoke_group = key_revoke.add_mutually_exclusive_group(required=True)
    key_revoke_group.add_argument("--user-id", type=int, help="User ID")
    key_revoke_group.add_argument("--email", help="User email")
    key_revoke.set_defaults(func=_admin_key_revoke)

    key_list = key_sub.add_parser("list", help="List registered users")
    key_list.add_argument("--limit", type=int, default=50)
    key_list.add_argument("--offset", type=int, default=0)
    key_list.set_defaults(func=_admin_key_list)

    # admin plan
    plan_p = admin_sub.add_parser("plan", help="Manage user plans")
    plan_sub = plan_p.add_subparsers(dest="plan_command", required=True)

    plan_set = plan_sub.add_parser("set", help="Set a user's plan")
    plan_set_user = plan_set.add_mutually_exclusive_group(required=True)
    plan_set_user.add_argument("--user-id", type=int, help="User ID")
    plan_set_user.add_argument("--email", help="User email")
    plan_set.add_argument("--plan", required=True, help="Plan key (free/pro/team/enterprise)")
    plan_set.set_defaults(func=_admin_plan_set)

    plan_info = plan_sub.add_parser("info", help="Show a user's plan and usage")
    plan_info_user = plan_info.add_mutually_exclusive_group(required=True)
    plan_info_user.add_argument("--user-id", type=int, help="User ID")
    plan_info_user.add_argument("--email", help="User email")
    plan_info.set_defaults(func=_admin_plan_info)

    # admin stats
    stats_p = admin_sub.add_parser("stats", help="Show aggregate usage statistics")
    stats_p.set_defaults(func=_admin_stats)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""
pipeline.py
===========
LLM EvalForge — main CI/CD eval runner.

Usage:
    python pipeline/pipeline.py [--model mock|simple] [--sample N] [--output path]

This script:
  1. Loads the golden dataset
  2. Runs the LLM against every (or a sample of) questions
  3. Measures hallucination, relevancy, faithfulness, latency, cost
  4. Decides whether to PASS or BLOCK the merge
  5. Writes a JSON report and a human-readable summary

Fixes applied vs original:
  - Removed unused 'time' import (F401)
  - Removed duplicate 'from typing import Optional' inside __main__ block
  - Fixed deprecated datetime.utcnow() → datetime.now(timezone.utc)
  - Moved sys.path hack above local imports to satisfy E402
  - Fixed all lines > 100 chars (E501)
"""

import json
import random
import argparse
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Make sure sibling packages are importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluators.evaluators import (  # noqa: E402
    evaluate_single,
    aggregate_results,
    LatencyTimer,
    EvalResult,
    AggregateMetrics,
    DEFAULT_THRESHOLDS,
)
from pipeline.model_adapters import get_model_adapter  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  ✓ Loaded {len(data)} questions from golden dataset")
    return data


def print_section(title: str):
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def print_result_row(r: EvalResult, idx: int):
    status = "✅ PASS" if r.passed else "❌ FAIL"
    print(
        f"  [{idx:3d}] {status}  Q:{r.question_id}"
        f"  rel={r.relevancy_score:.2f}"
        f"  hal={r.hallucination_score:.2f}"
        f"  sim={r.semantic_similarity:.2f}"
        f"  lat={r.latency_ms:.0f}ms"
    )
    for reason in r.failure_reasons:
        print(f"         ⚠ {reason}")


def print_aggregate_summary(agg: AggregateMetrics):
    print_section("📊  PIPELINE EVALUATION SUMMARY")
    print(f"  Total questions  : {agg.total_questions}")
    print(f"  Passed           : {agg.passed}  ({agg.pass_rate:.1%})")
    print(f"  Failed           : {agg.failed}")
    print()
    print(f"  Avg Relevancy    : {agg.avg_relevancy:.3f}")
    print(f"  Avg Faithfulness : {agg.avg_faithfulness:.3f}")
    print(f"  Hallucination    : {agg.avg_hallucination_rate:.3f}  (lower is better)")
    print(f"  Avg Similarity   : {agg.avg_semantic_similarity:.3f}")
    print(f"  Avg Completeness : {agg.avg_completeness:.3f}")
    print()
    print(f"  Latency p50      : {agg.latency_p50_ms:.1f} ms")
    print(f"  Latency p95      : {agg.latency_p95_ms:.1f} ms")
    print(f"  Cost / query     : ${agg.cost_per_query_usd:.5f}")
    print()

    if agg.blocked:
        print("  ╔══════════════════════════════════════════════╗")
        print("  ║  🚫  MERGE BLOCKED — quality gates failed   ║")
        print("  ╚══════════════════════════════════════════════╝")
        for reason in agg.block_reasons:
            print(f"    • {reason}")
    else:
        print("  ╔══════════════════════════════════════════════╗")
        print("  ║  ✅  ALL GATES PASSED — merge is approved   ║")
        print("  ╚══════════════════════════════════════════════╝")


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    dataset_path: str = "data/golden_dataset.json",
    model_name: str = "mock",
    sample_size: Optional[int] = None,
    output_dir: str = "reports",
    thresholds: Optional[dict] = None,
    verbose: bool = True,
) -> tuple[list[EvalResult], AggregateMetrics]:
    """
    Run the full eval pipeline.

    Returns:
        (results, aggregate_metrics)
    """
    # BUG FIX: datetime.utcnow() is deprecated since Python 3.12.
    # Use timezone-aware datetime.now(timezone.utc) instead.
    run_start = datetime.now(timezone.utc)

    if verbose:
        print_section(f"🚀  LLM EVALFORGE CI/CD PIPELINE  —  model: {model_name}")
        print(f"  Run started at {run_start.isoformat()}")

    # 1. Load dataset
    dataset = load_dataset(dataset_path)

    if sample_size and sample_size < len(dataset):
        dataset = random.sample(dataset, sample_size)
        if verbose:
            print(f"  ✓ Sampled {sample_size} questions for this run")

    # 2. Get the model adapter
    model = get_model_adapter(model_name)
    if verbose:
        print(f"  ✓ Using model adapter: {model.name}")

    # 3. Evaluate each question
    results: list[EvalResult] = []

    if verbose:
        print_section(f"🔍  Running {len(dataset)} evaluations")

    for idx, item in enumerate(dataset, start=1):
        context = item.get("context", "")

        with LatencyTimer() as timer:
            generated = model.generate(item["question"], context=context)

        result = evaluate_single(
            question_id=item["id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            generated_answer=generated,
            context=context,
            latency_ms=timer.elapsed_ms,
            thresholds=thresholds,
        )
        results.append(result)

        if verbose:
            print_result_row(result, idx)

    # 4. Aggregate
    agg = aggregate_results(results, thresholds=thresholds)

    if verbose:
        print_aggregate_summary(agg)

    # 5. Save report
    os.makedirs(output_dir, exist_ok=True)
    timestamp = run_start.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"eval_report_{timestamp}.json")

    report = {
        "meta": {
            "run_id": timestamp,
            "model": model_name,
            "run_at": run_start.isoformat(),
            "dataset_size": len(results),
            "thresholds": thresholds or DEFAULT_THRESHOLDS,
        },
        "aggregate": agg.to_dict(),
        "results": [r.to_dict() for r in results],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"\n  📄  Report saved → {report_path}")

    return results, agg


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM EvalForge CI/CD Pipeline")
    parser.add_argument(
        "--model",
        default="mock",
        help="Model adapter to use (mock | simple | openai | anthropic)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of questions to sample",
    )
    parser.add_argument(
        "--output",
        default="reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--dataset",
        default="data/golden_dataset.json",
        help="Path to golden dataset",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    # Resolve dataset path relative to project root
    root = Path(__file__).parent.parent
    dataset_path = root / args.dataset
    output_dir = root / args.output

    _, agg = run_pipeline(
        dataset_path=str(dataset_path),
        model_name=args.model,
        sample_size=args.sample,
        output_dir=str(output_dir),
        verbose=not args.quiet,
    )

    # Exit code: 1 if merge should be blocked
    sys.exit(1 if agg.blocked else 0)


if __name__ == "__main__":
    # BUG FIX: removed duplicate 'from typing import Optional' that was
    # erroneously placed here in the original code.
    main()

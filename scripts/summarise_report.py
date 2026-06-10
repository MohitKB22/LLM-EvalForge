#!/usr/bin/env python3
"""
scripts/summarise_report.py
============================
Parses the latest eval report and prints a compact human-readable summary.
Used by the CI workflow to annotate the run.

Fixes applied vs original:
  - Removed unused 'os' import (F401)
  - Fixed alignment spacing (E221)
  - Guard against questions shorter than 60 chars (IndexError on truncation)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"


def find_latest_report() -> Path | None:
    reports = sorted(REPORTS_DIR.glob("eval_report_*.json"))
    return reports[-1] if reports else None


def fmt(v: float, pct: bool = True) -> str:
    return f"{v*100:.1f}%" if pct else f"{v:.4f}"


def main():
    report_path = find_latest_report()
    if not report_path:
        print("⚠ No eval report found in reports/")
        sys.exit(0)

    with open(report_path) as f:
        report = json.load(f)

    meta = report["meta"]
    agg = report["aggregate"]
    results = report["results"]

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  LLM EvalForge Summary  |  model: {meta['model']}")
    print(f"  Run: {meta['run_at']}")
    print(sep)
    print(f"  Pass Rate     : {fmt(agg['pass_rate'])}  ({agg['passed']}/{agg['total_questions']})")
    print(f"  Hallucination : {fmt(agg['avg_hallucination_rate'])}  (lower is better)")
    print(f"  Relevancy     : {fmt(agg['avg_relevancy'])}")
    print(f"  Faithfulness  : {fmt(agg['avg_faithfulness'])}")
    print(f"  Similarity    : {fmt(agg['avg_semantic_similarity'])}")
    print(f"  Completeness  : {fmt(agg['avg_completeness'])}")
    print(f"  Latency p50   : {agg['latency_p50_ms']:.0f} ms")
    print(f"  Latency p95   : {agg['latency_p95_ms']:.0f} ms")
    print(f"  Cost/query    : ${agg['cost_per_query_usd']:.5f}")
    print()

    # Failure breakdown
    failed_items = [r for r in results if not r["passed"]]
    if failed_items:
        print(f"  ❌ Failed questions ({len(failed_items)}):")
        for r in failed_items[:10]:  # show first 10
            # BUG FIX: original code always appended '...' even for short questions.
            # Now only truncate if the question is actually long.
            question_preview = r["question"]
            if len(question_preview) > 60:
                question_preview = question_preview[:60] + "..."
            print(f"     • [{r['question_id']}] {question_preview}")
            for reason in r["failure_reasons"]:
                print(f"       ↳ {reason}")

    print()
    if agg["blocked"]:
        print("  ╔══════════════════════════════════════════╗")
        print("  ║  🚫  MERGE BLOCKED                      ║")
        print("  ╚══════════════════════════════════════════╝")
        for reason in agg["block_reasons"]:
            print(f"    • {reason}")
        sys.exit(1)
    else:
        print("  ╔══════════════════════════════════════════╗")
        print("  ║  ✅  ALL GATES PASSED — APPROVED        ║")
        print("  ╚══════════════════════════════════════════╝")
        sys.exit(0)


if __name__ == "__main__":
    main()

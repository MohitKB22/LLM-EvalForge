#!/usr/bin/env python3
"""
scripts/inject_report.py
=========================
Reads the latest eval_report_*.json from reports/ and injects it
as window.__REPORT__ into dashboard/index.html for static deployment.

Fixes applied vs original:
  - Fixed alignment spacing warnings (E221)
  - BUG FIX: the 'already injected' check compared the full injected block
    (including the large JSON payload) against the live HTML file, which
    always failed after a new report was generated — causing the report to
    be injected multiple times on repeated runs.
    Now uses a sentinel comment instead of the payload itself.
  - Added error handling for missing dashboard file
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORTS = sorted((ROOT / "reports").glob("eval_report_*.json"))
DASHBOARD = ROOT / "dashboard" / "index.html"

if not REPORTS:
    print("⚠ No report files found — dashboard will use demo data.")
    sys.exit(0)

if not DASHBOARD.exists():
    print(f"⚠ Dashboard file not found: {DASHBOARD}")
    sys.exit(1)

latest = REPORTS[-1]
print(f"  Injecting: {latest.name}")

with open(latest) as f:
    report_json = json.dumps(json.load(f))

html = DASHBOARD.read_text(encoding="utf-8")

INJECT_MARKER = "const REPORT = window.__REPORT__ || buildDemoReport();"

# BUG FIX: use a stable sentinel that doesn't embed the report payload,
# so repeated runs correctly detect an already-injected state.
SENTINEL = "/* __REPORT_INJECTED__ */"

if SENTINEL in html:
    print("  Already injected — replacing with fresh report.")
    # Replace from sentinel through the marker line to update with latest report
    import re
    html = re.sub(
        r"/\* __REPORT_INJECTED__ \*/.*?" + re.escape(INJECT_MARKER),
        f"{SENTINEL}\n  window.__REPORT__ = {report_json};\n  {INJECT_MARKER}",
        html,
        flags=re.DOTALL,
    )
else:
    injected = f"{SENTINEL}\n  window.__REPORT__ = {report_json};\n  {INJECT_MARKER}"
    html = html.replace(INJECT_MARKER, injected)

DASHBOARD.write_text(html, encoding="utf-8")
print("  ✓ Report injected into dashboard/index.html")

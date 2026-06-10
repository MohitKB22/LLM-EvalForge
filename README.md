# 🔬 LLM EvalForge

**A production-ready CI/CD pipeline for evaluating LLM quality gates — zero external ML dependencies.**

LLM EvalForge automatically measures hallucination rate, answer relevancy, faithfulness, semantic similarity, latency, and cost on every pull request. Merges are blocked when quality gates fail.

---

## Features

- **5 evaluation metrics** — relevancy, faithfulness, hallucination, semantic similarity, completeness
- **Zero external ML deps** — pure Python stdlib; no PyTorch, no sentence-transformers
- **Pluggable model adapters** — mock (CI), rule-based, OpenAI, Anthropic Claude
- **Configurable quality gates** — block merges when thresholds are breached
- **Rich CI/CD workflow** — lint → unit tests → eval pipeline → dashboard deploy
- **Live dashboard** — static HTML dashboard auto-deployed to GitHub Pages
- **102-question golden dataset** — multi-category, multi-difficulty, ready to use

---

## Project Structure

```
llm-evalforge/
├── evaluators/
│   ├── __init__.py
│   └── evaluators.py        # Core metrics: relevancy, faithfulness, hallucination, ...
├── pipeline/
│   ├── __init__.py
│   ├── model_adapters.py    # Mock, Simple, OpenAI, Anthropic adapters
│   └── pipeline.py          # Main runner — loads dataset, runs eval, saves report
├── tests/
│   └── test_evaluators.py   # 80 unit + integration tests
├── data/
│   └── golden_dataset.json  # 102 labelled QA pairs (geography, science, CS, ...)
├── scripts/
│   ├── summarise_report.py  # Pretty-print latest report (used in CI)
│   └── inject_report.py     # Inject report into dashboard for static deploy
├── dashboard/
│   └── index.html           # Self-contained dashboard (no build step)
├── reports/                 # Auto-generated eval reports (gitignored in prod)
├── .github/workflows/
│   └── eval_pipeline.yml    # Full CI/CD workflow
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline (mock adapter, full dataset)
python pipeline/pipeline.py

# 3. Run with sampling (faster)
python pipeline/pipeline.py --model mock --sample 20

# 4. Run unit tests
python -m pytest tests/ -v

# 5. View latest report summary
python scripts/summarise_report.py
```

---

## Model Adapters

| Adapter | Description | Requires |
|---|---|---|
| `mock` | Deterministic fake LLM, seeded RNG | Nothing |
| `simple` | Rule-based smart mock with pattern matching | Nothing |
| `openai` | Real OpenAI GPT | `pip install openai` + `OPENAI_API_KEY` |
| `anthropic` | Real Anthropic Claude | `pip install anthropic` + `ANTHROPIC_API_KEY` |

```bash
# Use OpenAI
export OPENAI_API_KEY=sk-...
python pipeline/pipeline.py --model openai

# Use Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-...
python pipeline/pipeline.py --model anthropic
```

---

## Quality Gates

The pipeline blocks a merge when any of these thresholds are breached:

| Gate | Default | Direction |
|---|---|---|
| Min relevancy | 0.35 | ↑ higher is better |
| Min faithfulness | 0.35 | ↑ higher is better |
| Max hallucination rate | 0.45 | ↓ lower is better |
| Min semantic similarity | 0.25 | ↑ higher is better |
| Min completeness | 0.20 | ↑ higher is better |
| Max p95 latency | 15,000 ms | ↓ lower is better |
| Min pass rate | 65% | ↑ higher is better |

Override thresholds via the `thresholds` argument in `run_pipeline()` or by modifying `DEFAULT_THRESHOLDS`.

---

## CI/CD Workflow

The GitHub Actions workflow runs 4 jobs on every push / PR:

1. **Lint & Style** — `black`, `isort`, `flake8`
2. **Unit Tests** — pytest across Python 3.10, 3.11, 3.12 with ≥75% coverage gate
3. **Eval Pipeline** — full quality gate run; posts results as PR comment; fails CI if blocked
4. **Deploy Dashboard** — injects latest report into `dashboard/index.html` and deploys to GitHub Pages (main branch only)

### Required Secrets

| Secret | Purpose |
|---|---|
| `OPENAI_API_KEY` | Optional — only needed with `--model openai` |
| `ANTHROPIC_API_KEY` | Optional — only needed with `--model anthropic` |

---

## Metrics Reference

### Semantic Similarity
TF-IDF-style Jaccard + BLEU-1 blend over content tokens (stopwords removed). Range: 0–1.

### Relevancy
Measures content-word coverage of the question in the answer, with a short-answer length penalty. Range: 0–1.

### Faithfulness
Token overlap between the answer and provided context (RAG grounding). Returns 0.7 when no context is supplied. Range: 0–1.

### Hallucination Score
Composite of semantic similarity, context faithfulness, trigram overlap, and negation penalty. **Lower is better.** Range: 0–1.

### Completeness
Bigram recall of the expected answer's key content. Range: 0–1.

---

## Bug Fixes (vs original `llm-eval-cicd`)

| File | Bug | Fix |
|---|---|---|
| `evaluators/evaluators.py` | Unused `math` import | Removed |
| `evaluators/evaluators.py` | `expected_tokens` assigned but never read | Removed dead assignment |
| `pipeline/model_adapters.py` | Duplicate dict key `"photosynthesis"` — second entry silently overwrote first | Merged into single comprehensive entry |
| `pipeline/model_adapters.py` | Unused `typing.Optional` import | Removed |
| `pipeline/pipeline.py` | Unused `time` import | Removed |
| `pipeline/pipeline.py` | Duplicate `from typing import Optional` inside `__main__` block | Removed duplicate |
| `pipeline/pipeline.py` | `datetime.utcnow()` deprecated since Python 3.12 | Replaced with `datetime.now(timezone.utc)` |
| `scripts/inject_report.py` | `already injected` check compared full payload — always re-injected on new reports | Replaced with stable `/* __REPORT_INJECTED__ */` sentinel |
| `scripts/inject_report.py` | No error handling if dashboard file is missing | Added `DASHBOARD.exists()` guard |
| `scripts/summarise_report.py` | Unconditional `[:60]...` truncation on short questions | Only truncate when `len > 60` |
| `scripts/summarise_report.py` | Unused `os` import | Removed |
| `tests/test_evaluators.py` | Unused `os` and `math` imports | Removed |
| All files | Lines exceeding 100-char limit (E501) | Reformatted |
| All files | Alignment whitespace warnings (E221) | Fixed |

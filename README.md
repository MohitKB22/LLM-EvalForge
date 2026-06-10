# 🔬 LLM EvalForge

### Production-Ready CI/CD Quality Gates for Large Language Models

LLM EvalForge is a lightweight evaluation framework that automatically measures LLM quality on every pull request and blocks merges when performance regresses.

Built for teams shipping AI products, EvalForge provides reproducible evaluation pipelines, configurable quality gates, automated reporting, and GitHub-native CI/CD integration — all with **zero heavyweight ML dependencies**.

---

## ✨ Highlights

* 📊 **5 Core Evaluation Metrics**

  * Relevancy
  * Faithfulness
  * Hallucination Rate
  * Semantic Similarity
  * Completeness

* 🚀 **CI/CD First**

  * Automated evaluation on every push and pull request
  * Merge blocking through configurable quality gates
  * GitHub Actions integration out of the box

* 🪶 **Lightweight by Design**

  * Pure Python implementation
  * No PyTorch
  * No Transformers
  * No Sentence-Transformers

* 🔌 **Pluggable Model Adapters**

  * Mock adapter for deterministic CI runs
  * Rule-based adapter
  * OpenAI GPT models
  * Anthropic Claude models

* 📈 **Live Evaluation Dashboard**

  * Static HTML dashboard
  * GitHub Pages deployment
  * No frontend build step required

* 🧪 **Golden Dataset Included**

  * 102 curated evaluation questions
  * Multiple domains and difficulty levels
  * Ready for benchmarking and regression testing

---

## 🎯 Why EvalForge?

Most LLM evaluation frameworks require heavyweight ML dependencies, GPU resources, or complex infrastructure.

EvalForge focuses on a different goal:

> **Fast, deterministic, CI-friendly quality checks that can run on every pull request.**

This makes it ideal for:

* AI-powered applications
* Retrieval-Augmented Generation (RAG) systems
* Internal copilots and assistants
* Customer support bots
* LLM experimentation pipelines
* AI engineering portfolios
* 
---

## 🤖 Supported Model Adapters

| Adapter     | Description                           | Requirements |
| ----------- | ------------------------------------- | ------------ |
| `mock`      | Deterministic fake LLM for CI testing | None         |
| `simple`    | Rule-based answer generator           | None         |
| `openai`    | OpenAI GPT models                     | API Key      |
| `anthropic` | Anthropic Claude models               | API Key      |

---

## 🔄 CI/CD Workflow

Every push or pull request triggers:

1. 🎨 Lint & Style Validation
2. 🧪 Unit Testing
3. 📊 LLM Evaluation Pipeline
4. 🚀 Dashboard Deployment

The workflow can:

* Fail CI when quality gates are breached
* Generate evaluation reports
* Post evaluation summaries to pull requests
* Publish dashboards via GitHub Pages

---

## 📚 Evaluation Metrics

### Semantic Similarity

Measures lexical and content similarity using a TF-IDF-inspired Jaccard + BLEU blend.

### Relevancy

Evaluates how well an answer addresses the original question.

### Faithfulness

Measures grounding against supplied context.

### Hallucination Score

Detects unsupported or fabricated content.

### Completeness

Measures coverage of expected answer content.

---

## 🛠️ Improvements Over `llm-eval-cicd`

This project includes multiple fixes and refactors:

* ✅ Removed unused imports and dead code
* ✅ Fixed duplicate dictionary key overwrite bug
* ✅ Replaced deprecated Python 3.12 APIs
* ✅ Improved dashboard injection reliability
* ✅ Added defensive error handling
* ✅ Fixed formatting and lint violations
* ✅ Improved report summarisation logic

---

## 📝 License

MIT License

---

## ⭐ If you find this project useful

Give it a star and feel free to fork it for your own LLM evaluation workflows.

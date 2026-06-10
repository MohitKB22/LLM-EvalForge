"""
evaluators.py
=============
Core evaluation metrics for LLM EvalForge CI/CD Pipeline.
Measures: hallucination rate, answer relevancy, faithfulness, latency.

Fixes applied vs original:
  - Removed unused 'math' import (F401)
  - Removed unused local variable 'expected_tokens' (F841)
  - Fixed all lines > 100 chars (E501)
"""

import time
import re
from typing import Optional
from collections import Counter
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Holds all scores for a single question-answer pair."""
    question_id: str
    question: str
    expected_answer: str
    generated_answer: str
    context: str = ""
    latency_ms: float = 0.0

    # Scores (all 0‒1, higher = better, except hallucination lower = better)
    relevancy_score: float = 0.0
    faithfulness_score: float = 0.0
    hallucination_score: float = 0.0  # probability the answer contains a hallucination
    semantic_similarity: float = 0.0
    completeness_score: float = 0.0

    passed: bool = False
    failure_reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "context": self.context,
            "latency_ms": round(self.latency_ms, 2),
            "scores": {
                "relevancy": round(self.relevancy_score, 4),
                "faithfulness": round(self.faithfulness_score, 4),
                "hallucination": round(self.hallucination_score, 4),
                "semantic_similarity": round(self.semantic_similarity, 4),
                "completeness": round(self.completeness_score, 4),
            },
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
        }


@dataclass
class AggregateMetrics:
    """Summary statistics across all evaluated pairs."""
    total_questions: int = 0
    passed: int = 0
    failed: int = 0

    avg_relevancy: float = 0.0
    avg_faithfulness: float = 0.0
    avg_hallucination_rate: float = 0.0
    avg_semantic_similarity: float = 0.0
    avg_completeness: float = 0.0

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    cost_per_query_usd: float = 0.0

    pass_rate: float = 0.0
    blocked: bool = False
    block_reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_questions": self.total_questions,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "avg_relevancy": round(self.avg_relevancy, 4),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_hallucination_rate": round(self.avg_hallucination_rate, 4),
            "avg_semantic_similarity": round(self.avg_semantic_similarity, 4),
            "avg_completeness": round(self.avg_completeness, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "cost_per_query_usd": round(self.cost_per_query_usd, 6),
            "blocked": self.blocked,
            "block_reasons": self.block_reasons,
        }


# ─────────────────────────────────────────────────────────────
# Tokenisation helpers (no external deps)
# ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


# ─────────────────────────────────────────────────────────────
# Individual metric functions
# ─────────────────────────────────────────────────────────────

def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Approximate cosine similarity using TF-IDF-style weighted token overlap.
    Scores 0‒1 (1 = identical meaning).
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))

    if not tokens_a or not tokens_b:
        return 0.0

    # Stopwords (common English words that carry little meaning)
    stopwords = {
        "the", "a", "an", "is", "it", "in", "of", "to", "and", "or",
        "that", "this", "for", "with", "on", "at", "by", "from", "are",
        "was", "were", "be", "been", "has", "have", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "can",
        "not", "but", "as", "if", "then", "than", "so", "up", "out",
        "its", "their", "they", "we", "you", "he", "she", "i", "me",
        "him", "her", "them", "us", "my", "your", "our", "their", "also",
        "which", "what", "who", "how", "when", "where", "all", "more",
        "some", "any", "each", "such", "into", "about", "very", "just",
    }

    tokens_a = {t for t in tokens_a if t not in stopwords and len(t) > 1}
    tokens_b = {t for t in tokens_b if t not in stopwords and len(t) > 1}

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = intersection / union if union > 0 else 0.0

    # BLEU-1 style precision component
    bleu = intersection / len(tokens_a) if tokens_a else 0.0

    # Blend
    return round(0.6 * jaccard + 0.4 * bleu, 4)


def compute_relevancy(question: str, generated_answer: str) -> float:
    """
    Measures how well the answer addresses the question.
    Checks key question words appear contextually in the answer.
    """
    if not generated_answer.strip():
        return 0.0

    q_tokens = set(_tokenize(question))
    a_tokens = set(_tokenize(generated_answer))

    # Filter to content words (length > 3, not stopword-like)
    question_content = {t for t in q_tokens if len(t) > 3}

    if not question_content:
        return compute_semantic_similarity(question, generated_answer)

    coverage = len(question_content & a_tokens) / len(question_content)

    # Length sanity: very short answers are penalised
    answer_words = len(_tokenize(generated_answer))
    length_penalty = min(1.0, answer_words / 20)

    return round(0.7 * coverage + 0.3 * length_penalty, 4)


def compute_faithfulness(generated_answer: str, context: str) -> float:
    """
    Measures whether the answer is grounded in the provided context.
    If no context is provided returns a neutral 0.7 (no RAG scenario).
    """
    if not context.strip():
        return 0.7  # no context supplied → can't penalise

    if not generated_answer.strip():
        return 0.0

    a_tokens = _tokenize(generated_answer)
    c_tokens = set(_tokenize(context))

    if not a_tokens:
        return 0.0

    grounded = sum(1 for t in a_tokens if t in c_tokens)
    return round(grounded / len(a_tokens), 4)


def compute_hallucination_score(
    generated_answer: str,
    expected_answer: str,
    context: str = "",
) -> float:
    """
    Estimates hallucination probability (0 = no hallucination, 1 = full).

    Logic:
      1. If the answer contradicts key facts in expected → high score
      2. If the answer contains statements not supported by context → medium
      3. Cross-check using trigram overlap
    """
    if not generated_answer.strip():
        return 1.0

    similarity = compute_semantic_similarity(generated_answer, expected_answer)

    # Contradiction heuristic: look for negation of key expected terms
    # BUG FIX: 'expected_tokens' was assigned but never used — removed
    gen_lower = generated_answer.lower()

    contradiction_signals = ["not", "never", "no ", "false", "incorrect", "wrong"]
    contradiction_count = sum(1 for sig in contradiction_signals if sig in gen_lower)
    contradiction_penalty = min(0.3, contradiction_count * 0.05)

    # Trigram overlap between generated and expected
    exp_tri = _ngrams(_tokenize(expected_answer), 3)
    gen_tri = _ngrams(_tokenize(generated_answer), 3)
    if exp_tri:
        tri_overlap = (
            sum(min(gen_tri[g], exp_tri[g]) for g in gen_tri) / sum(exp_tri.values())
        )
    else:
        tri_overlap = 0.0

    # Context faithfulness component
    if context.strip():
        faith = compute_faithfulness(generated_answer, context)
        hallucination = 1 - (0.4 * similarity + 0.4 * faith + 0.2 * tri_overlap)
    else:
        hallucination = 1 - (0.6 * similarity + 0.4 * tri_overlap)

    hallucination = max(0.0, hallucination + contradiction_penalty)
    return round(min(1.0, hallucination), 4)


def compute_completeness(generated_answer: str, expected_answer: str) -> float:
    """
    Measures how much of the expected answer's key content is covered.
    Uses bigram recall.
    """
    if not expected_answer.strip() or not generated_answer.strip():
        return 0.0

    exp_bi = _ngrams(_tokenize(expected_answer), 2)
    gen_bi = _ngrams(_tokenize(generated_answer), 2)

    if not exp_bi:
        return 1.0

    recall = sum(min(gen_bi[g], exp_bi[g]) for g in exp_bi) / sum(exp_bi.values())
    return round(recall, 4)


# ─────────────────────────────────────────────────────────────
# Latency tracking
# ─────────────────────────────────────────────────────────────

class LatencyTimer:
    """Simple context manager for measuring wall-clock latency in ms."""

    def __init__(self):
        self._start = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def percentile(values: list[float], pct: float) -> float:
    """Return the p-th percentile of a sorted list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (pct / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ─────────────────────────────────────────────────────────────
# Thresholds & pass/fail logic
# ─────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "min_relevancy": 0.35,
    "min_faithfulness": 0.35,
    "max_hallucination_rate": 0.45,  # block merge if > 45 %
    "min_semantic_similarity": 0.25,
    "min_completeness": 0.20,
    "max_latency_p95_ms": 15_000,  # 15 s SLA
    "min_pass_rate": 0.65,  # 65 % of questions must pass
}


def evaluate_single(
    question_id: str,
    question: str,
    expected_answer: str,
    generated_answer: str,
    context: str = "",
    latency_ms: float = 0.0,
    thresholds: Optional[dict] = None,
) -> EvalResult:
    """Evaluate one QA pair and return an EvalResult."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    result = EvalResult(
        question_id=question_id,
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        context=context,
        latency_ms=latency_ms,
    )

    result.relevancy_score = compute_relevancy(question, generated_answer)
    result.faithfulness_score = compute_faithfulness(generated_answer, context)
    result.hallucination_score = compute_hallucination_score(
        generated_answer, expected_answer, context
    )
    result.semantic_similarity = compute_semantic_similarity(
        generated_answer, expected_answer
    )
    result.completeness_score = compute_completeness(generated_answer, expected_answer)

    # Pass/fail per question
    failures = []
    if result.relevancy_score < t["min_relevancy"]:
        failures.append(
            f"relevancy {result.relevancy_score:.2f} < {t['min_relevancy']}"
        )
    if result.faithfulness_score < t["min_faithfulness"]:
        failures.append(
            f"faithfulness {result.faithfulness_score:.2f} < {t['min_faithfulness']}"
        )
    if result.hallucination_score > t["max_hallucination_rate"]:
        failures.append(
            f"hallucination {result.hallucination_score:.2f} > {t['max_hallucination_rate']}"
        )
    if result.semantic_similarity < t["min_semantic_similarity"]:
        failures.append(
            f"similarity {result.semantic_similarity:.2f} < {t['min_semantic_similarity']}"
        )
    if result.completeness_score < t["min_completeness"]:
        failures.append(
            f"completeness {result.completeness_score:.2f} < {t['min_completeness']}"
        )
    if latency_ms > t["max_latency_p95_ms"]:
        failures.append(
            f"latency {latency_ms:.0f}ms > {t['max_latency_p95_ms']}ms SLA"
        )

    result.failure_reasons = failures
    result.passed = len(failures) == 0
    return result


def aggregate_results(
    results: list[EvalResult],
    thresholds: Optional[dict] = None,
    cost_per_1k_tokens: float = 0.003,
    avg_tokens_per_query: int = 300,
) -> AggregateMetrics:
    """Aggregate per-question results into pipeline-level metrics."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    n = len(results)
    if n == 0:
        return AggregateMetrics()

    agg = AggregateMetrics()
    agg.total_questions = n
    agg.passed = sum(1 for r in results if r.passed)
    agg.failed = n - agg.passed
    agg.pass_rate = agg.passed / n

    agg.avg_relevancy = sum(r.relevancy_score for r in results) / n
    agg.avg_faithfulness = sum(r.faithfulness_score for r in results) / n
    agg.avg_hallucination_rate = sum(r.hallucination_score for r in results) / n
    agg.avg_semantic_similarity = sum(r.semantic_similarity for r in results) / n
    agg.avg_completeness = sum(r.completeness_score for r in results) / n

    latencies = [r.latency_ms for r in results]
    agg.latency_p50_ms = percentile(latencies, 50)
    agg.latency_p95_ms = percentile(latencies, 95)

    agg.cost_per_query_usd = (avg_tokens_per_query / 1000) * cost_per_1k_tokens

    # Pipeline-level block decision
    block_reasons = []
    if agg.avg_hallucination_rate > t["max_hallucination_rate"]:
        block_reasons.append(
            f"Hallucination rate {agg.avg_hallucination_rate:.1%} > "
            f"threshold {t['max_hallucination_rate']:.1%}"
        )
    if agg.latency_p95_ms > t["max_latency_p95_ms"]:
        block_reasons.append(
            f"p95 latency {agg.latency_p95_ms:.0f}ms exceeds SLA "
            f"{t['max_latency_p95_ms']}ms"
        )
    if agg.pass_rate < t["min_pass_rate"]:
        block_reasons.append(
            f"Pass rate {agg.pass_rate:.1%} < minimum {t['min_pass_rate']:.1%}"
        )

    agg.blocked = len(block_reasons) > 0
    agg.block_reasons = block_reasons
    return agg

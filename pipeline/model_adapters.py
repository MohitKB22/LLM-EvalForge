"""
model_adapters.py
=================
Pluggable model adapters for the LLM EvalForge pipeline.

Available adapters
------------------
  mock      — Deterministic fake answers, fast, reproducible. Great for CI testing.
  simple    — Rule-based "smart mock" that answers factual questions reasonably well.
  openai    — Real OpenAI API (requires OPENAI_API_KEY env var).
  anthropic — Real Anthropic Claude API (requires ANTHROPIC_API_KEY env var).

Usage:
    adapter = get_model_adapter("mock")
    answer  = adapter.generate("What is Python?")

Fixes applied vs original:
  - Removed unused 'typing.Optional' import (F401)
  - Fixed duplicate dict key 'photosynthesis' (F601) — merged into one entry
  - Fixed all lines > 100 chars (E501)
"""

import os
import random
import time
import re
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────

class ModelAdapter(ABC):
    """Abstract base for any LLM adapter used in the eval pipeline."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(self, question: str, context: str = "") -> str:
        """Generate an answer for the given question, optionally grounded in context."""
        ...


# ─────────────────────────────────────────────────────────────
# Mock adapter — fast, deterministic, no external calls
# ─────────────────────────────────────────────────────────────

# BUG FIX: duplicate key 'photosynthesis' — the second entry silently overwrote
# the first. Merged into a single, more complete definition.
_MOCK_KNOWLEDGE: dict[str, str] = {
    "capital of france": "Paris is the capital of France.",
    "photosynthesis": (
        "Photosynthesis is the process by which plants use sunlight, water, "
        "and CO₂ to produce glucose and oxygen, converting light energy into "
        "chemical energy stored in glucose."
    ),
    "pythagorean theorem": (
        "The Pythagorean theorem states that a² + b² = c² in a right triangle."
    ),
    "speed of light": (
        "The speed of light is approximately 299,792,458 meters per second."
    ),
    "what is an api": (
        "An API (Application Programming Interface) is a set of rules that "
        "allows software applications to communicate with each other."
    ),
    "machine learning": (
        "Machine learning is a subset of AI where computers learn patterns "
        "from data without being explicitly programmed."
    ),
    "world war ii": (
        "World War II ended in 1945 with the surrender of Germany and Japan."
    ),
    "dna": (
        "DNA (Deoxyribonucleic Acid) is a molecule containing the genetic "
        "instructions for development and functioning of all known living organisms."
    ),
    "black hole": (
        "A black hole is a region of spacetime where gravity is so strong that "
        "nothing can escape it, not even light."
    ),
    "what is sql": (
        "SQL (Structured Query Language) is used to manage and query relational databases."
    ),
    "rest api": (
        "A REST API uses HTTP methods to enable communication between software "
        "systems following REST architecture principles."
    ),
    "blockchain": (
        "A blockchain is a distributed ledger where transactions are recorded in "
        "linked blocks that are tamper-resistant."
    ),
    "neural network": (
        "A neural network is a computing model inspired by the brain, consisting "
        "of interconnected nodes that learn from data."
    ),
    "climate change": (
        "Climate change is primarily caused by human activities releasing greenhouse "
        "gases, especially burning fossil fuels."
    ),
    "hallucination": (
        "In AI, hallucination refers to when a language model generates confident "
        "but factually incorrect information."
    ),
    "rag": (
        "RAG (Retrieval-Augmented Generation) combines document retrieval with "
        "language model generation to produce grounded answers."
    ),
    "transformer": (
        "A transformer is a neural network architecture that uses self-attention "
        "mechanisms, forming the basis of modern LLMs."
    ),
    "overfitting": (
        "Overfitting occurs when a model learns training data too well, performing "
        "poorly on new, unseen data."
    ),
    "compound interest": (
        "Compound interest grows exponentially by calculating interest on both "
        "the principal and accumulated interest."
    ),
}

_GENERIC_ANSWERS = [
    "This is an interesting question. The answer depends on several factors "
    "that need to be carefully considered.",
    "Based on available information, there are multiple perspectives on this "
    "topic worth exploring.",
    "The concept you're asking about has several dimensions. Generally speaking, "
    "it involves understanding the underlying principles.",
    "This topic requires careful analysis. In essence, the key aspects include "
    "the fundamental principles and their applications.",
    "The answer to this question involves understanding several interconnected "
    "concepts and their relationships.",
]


class MockModelAdapter(ModelAdapter):
    """
    Deterministic mock adapter.
    Returns plausible answers for known questions,
    and generates slightly noisy generic answers otherwise —
    simulating occasional hallucinations for testing the evaluators.
    """

    def __init__(self, hallucination_rate: float = 0.15, seed: int = 42):
        self._rng = random.Random(seed)
        self._hallucination_rate = hallucination_rate

    @property
    def name(self) -> str:
        return "mock-llm"

    def generate(self, question: str, context: str = "") -> str:
        time.sleep(0.001)  # simulate minimal latency

        q_lower = question.lower()

        # Try to match known topics
        for key, answer in _MOCK_KNOWLEDGE.items():
            if key in q_lower:
                # Occasionally inject a hallucination
                if self._rng.random() < self._hallucination_rate:
                    return self._hallucinate(answer)
                return answer

        # Context-grounded response
        if context:
            context_words = context.split()[:30]
            return f"Based on the context: {' '.join(context_words)}..."

        # Generic fallback with occasional hallucination
        if self._rng.random() < self._hallucination_rate * 2:
            return self._rng.choice(
                [
                    "I'm not sure but I believe this is completely false and incorrect.",
                    "The answer is definitely 42 for all questions of this type.",
                    "This was invented by Napoleon in 1803.",
                ]
            )

        return self._rng.choice(_GENERIC_ANSWERS)

    def _hallucinate(self, correct_answer: str) -> str:
        """Insert a factual error into a correct answer."""
        corruptions = [
            (" is ", " is not "),
            ("primary", "secondary"),
            ("increases", "decreases"),
            ("positive", "negative"),
        ]
        answer = correct_answer
        for orig, replacement in self._rng.sample(corruptions, 1):
            if orig in answer:
                return answer.replace(orig, replacement, 1)
        return "This is actually completely different from what most people believe."


# ─────────────────────────────────────────────────────────────
# Simple rule-based adapter (smarter mock, still no API)
# ─────────────────────────────────────────────────────────────

class SimpleModelAdapter(ModelAdapter):
    """
    A smarter rule-based adapter that produces more realistic answers
    by composing knowledge fragments. Good for development/debugging.
    """

    @property
    def name(self) -> str:
        return "simple-rule-based"

    def generate(self, question: str, context: str = "") -> str:
        time.sleep(self._simulate_latency())
        q = question.lower().strip()

        # ── Direct knowledge ────────────────────────────────
        for key, answer in _MOCK_KNOWLEDGE.items():
            if key in q:
                return answer

        # ── Pattern matching ────────────────────────────────
        if re.search(r"\bwhat is\b|\bdefine\b|\bexplain\b", q):
            topic = re.sub(r"what is|define|explain|the|a |an ", "", q).strip()
            return (
                f"{topic.capitalize()} is a fundamental concept that involves "
                f"understanding the core principles and how they apply in practice. "
                f"It plays an important role in its respective field."
            )

        if re.search(r"\bhow does\b|\bhow do\b", q):
            return (
                "The process works by first identifying the inputs, then applying "
                "the relevant transformation or logic, and finally producing an output. "
                "Each step builds on the previous one."
            )

        if re.search(r"\bwhen\b|\bwhat year\b|\bwhat date\b", q):
            return (
                "This event occurred during a historically significant period "
                "and had lasting consequences."
            )

        if re.search(r"\bwho\b", q):
            return (
                "This was pioneered by several key individuals who made fundamental "
                "contributions to the field."
            )

        if re.search(r"\bdifference between\b|\bvs\b|\bversus\b", q):
            return (
                "The key differences lie in their purpose, implementation, and use cases. "
                "One approach is better suited for certain scenarios while the other "
                "excels in different contexts."
            )

        # Context-grounded
        if context:
            sentences = context.split(".")[:2]
            return "Based on the available information: " + ". ".join(
                s.strip() for s in sentences if s.strip()
            ) + "."

        return (
            "This is a complex topic with multiple dimensions that require "
            "careful consideration of the available evidence."
        )

    def _simulate_latency(self) -> float:
        """Realistic latency between 50‒800 ms."""
        return random.uniform(0.05, 0.8)


# ─────────────────────────────────────────────────────────────
# OpenAI adapter (real API — optional)
# ─────────────────────────────────────────────────────────────

class OpenAIModelAdapter(ModelAdapter):
    """
    Real OpenAI adapter. Requires:
      pip install openai
      export OPENAI_API_KEY=sk-...
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise EnvironmentError(
                        "OPENAI_API_KEY not set. Export it before using the openai adapter."
                    )
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client

    @property
    def name(self) -> str:
        return f"openai-{self._model}"

    def generate(self, question: str, context: str = "") -> str:
        client = self._get_client()
        system = (
            "You are a helpful, accurate assistant. "
            "Answer questions concisely and factually."
        )
        user_msg = f"Question: {question}"
        if context:
            user_msg = f"Context:\n{context}\n\n{user_msg}"

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# Anthropic adapter (real API — optional)
# ─────────────────────────────────────────────────────────────

class AnthropicModelAdapter(ModelAdapter):
    """
    Real Anthropic Claude adapter. Requires:
      pip install anthropic
      export ANTHROPIC_API_KEY=sk-ant-...
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise EnvironmentError(
                        "ANTHROPIC_API_KEY not set. "
                        "Export it before using the anthropic adapter."
                    )
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        return self._client

    @property
    def name(self) -> str:
        return f"anthropic-{self._model}"

    def generate(self, question: str, context: str = "") -> str:
        client = self._get_client()
        user_msg = f"Question: {question}"
        if context:
            user_msg = f"Context:\n{context}\n\n{user_msg}"

        response = client.messages.create(
            model=self._model,
            max_tokens=256,
            system=(
                "You are a helpful, accurate assistant. "
                "Answer questions concisely and factually."
            ),
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text.strip()


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

_ADAPTERS = {
    "mock": MockModelAdapter,
    "simple": SimpleModelAdapter,
    "openai": OpenAIModelAdapter,
    "anthropic": AnthropicModelAdapter,
}


def get_model_adapter(name: str, **kwargs) -> ModelAdapter:
    """
    Factory function — returns a ModelAdapter by name.

    Args:
        name: "mock" | "simple" | "openai" | "anthropic"
        **kwargs: Passed to the adapter constructor (e.g. model="gpt-4")
    """
    name = name.lower().strip()
    if name not in _ADAPTERS:
        raise ValueError(
            f"Unknown model adapter '{name}'. "
            f"Choose from: {', '.join(_ADAPTERS.keys())}"
        )
    return _ADAPTERS[name](**kwargs)

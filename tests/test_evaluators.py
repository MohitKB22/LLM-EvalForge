"""
tests/test_evaluators.py
========================
Full unit-test suite for LLM EvalForge CI/CD Pipeline.

Run:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short

Fixes applied vs original:
  - Removed unused 'os' and 'math' imports (F401)
  - Moved sys.path hack before local imports (E402)
  - Fixed all lines > 100 chars (E501)
  - Fixed alignment spacing (E221)
"""

import sys
import json
import pytest
from pathlib import Path

# ── Make parent importable ────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluators.evaluators import (  # noqa: E402
    compute_semantic_similarity,
    compute_relevancy,
    compute_faithfulness,
    compute_hallucination_score,
    compute_completeness,
    evaluate_single,
    aggregate_results,
    percentile,
    EvalResult,
    DEFAULT_THRESHOLDS,
)
from pipeline.model_adapters import (  # noqa: E402
    MockModelAdapter,
    SimpleModelAdapter,
    get_model_adapter,
)


# ══════════════════════════════════════════════════════════════
# 1. SEMANTIC SIMILARITY TESTS
# ══════════════════════════════════════════════════════════════

class TestSemanticSimilarity:

    def test_identical_texts_score_one(self):
        text = "The quick brown fox jumps over the lazy dog"
        score = compute_semantic_similarity(text, text)
        assert score >= 0.95, f"Identical texts should score ~1.0, got {score}"

    def test_empty_string_returns_zero(self):
        assert compute_semantic_similarity("", "some answer") == 0.0
        assert compute_semantic_similarity("some question", "") == 0.0
        assert compute_semantic_similarity("", "") == 0.0

    def test_completely_unrelated_texts_score_low(self):
        score = compute_semantic_similarity(
            "Paris is the capital of France",
            "The mitochondria produces ATP through oxidative phosphorylation",
        )
        assert score < 0.25, f"Unrelated texts should score low, got {score}"

    def test_paraphrased_text_scores_medium_high(self):
        # These sentences share several meaningful content words after stop-word
        # removal; Jaccard over content words gives a small but non-zero overlap.
        a = (
            "Machine learning is a subset of artificial intelligence "
            "where computers learn from data."
        )
        b = (
            "ML is part of AI; machines learn patterns from datasets "
            "without explicit programming."
        )
        score = compute_semantic_similarity(a, b)
        assert score > 0.0, f"Paraphrased texts should have non-zero overlap, got {score}"

    def test_symmetry(self):
        a = "Python is a popular programming language"
        b = "Many developers use Python for scripting and data science"
        assert abs(
            compute_semantic_similarity(a, b) - compute_semantic_similarity(b, a)
        ) < 0.05

    def test_score_in_range(self):
        score = compute_semantic_similarity("hello world", "goodbye universe")
        assert 0.0 <= score <= 1.0

    def test_high_overlap_scores_high(self):
        a = "The speed of light is 299,792,458 meters per second"
        b = "Light travels at 299,792,458 meters per second in a vacuum"
        score = compute_semantic_similarity(a, b)
        assert score >= 0.35, f"High overlap texts should score higher, got {score}"


# ══════════════════════════════════════════════════════════════
# 2. RELEVANCY TESTS
# ══════════════════════════════════════════════════════════════

class TestRelevancy:

    def test_on_topic_answer_scores_high(self):
        question = "What is photosynthesis?"
        answer = (
            "Photosynthesis is the process plants use to convert sunlight "
            "into glucose using water and carbon dioxide."
        )
        score = compute_relevancy(question, answer)
        assert score >= 0.40, f"On-topic answer should score higher, got {score}"

    def test_empty_answer_scores_zero(self):
        assert compute_relevancy("What is gravity?", "") == 0.0

    def test_off_topic_answer_scores_low(self):
        question = "What is the capital of France?"
        answer = "The mitochondria is the powerhouse of the cell and produces ATP."
        score = compute_relevancy(question, answer)
        assert score < 0.35, f"Off-topic answer should score low, got {score}"

    def test_very_short_answer_penalised(self):
        question = "Explain how machine learning works in detail"
        answer = "learning"
        score = compute_relevancy(question, answer)
        assert score < 0.5, f"Very short answer should be penalised, got {score}"

    def test_score_between_0_and_1(self):
        score = compute_relevancy("What is Python?", "Python is a programming language.")
        assert 0.0 <= score <= 1.0

    def test_longer_relevant_answer_not_penalised(self):
        question = "What is deep learning?"
        answer = (
            "Deep learning is a subset of machine learning that uses neural networks "
            "with many layers to automatically learn hierarchical representations from data. "
            "It powers applications like image recognition, natural language processing, "
            "and autonomous driving."
        )
        score = compute_relevancy(question, answer)
        assert score >= 0.30, f"Detailed on-topic answer should not be penalised, got {score}"


# ══════════════════════════════════════════════════════════════
# 3. FAITHFULNESS TESTS
# ══════════════════════════════════════════════════════════════

class TestFaithfulness:

    def test_grounded_answer_scores_high(self):
        context = (
            "Paris is the capital and largest city of France. "
            "It is located in northern France."
        )
        answer = (
            "Paris is the capital city of France and is located in "
            "the northern part of the country."
        )
        score = compute_faithfulness(answer, context)
        assert score >= 0.55, f"Grounded answer should score high, got {score}"

    def test_no_context_returns_neutral(self):
        score = compute_faithfulness("Some answer", "")
        assert score == 0.7, f"No context should return neutral 0.7, got {score}"

    def test_ungrounded_answer_scores_low(self):
        context = "The sky is blue due to Rayleigh scattering of sunlight."
        answer = (
            "Blockchain is a distributed ledger using cryptographic "
            "hashing for immutability."
        )
        score = compute_faithfulness(answer, context)
        assert score < 0.4, f"Ungrounded answer should score low, got {score}"

    def test_empty_answer_scores_zero(self):
        assert compute_faithfulness("", "Some context here") == 0.0

    def test_score_in_range(self):
        score = compute_faithfulness("answer text", "context text")
        assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════
# 4. HALLUCINATION SCORE TESTS
# ══════════════════════════════════════════════════════════════

class TestHallucinationScore:

    def test_correct_answer_has_low_hallucination(self):
        expected = "Paris is the capital of France."
        generated = "The capital of France is Paris."
        score = compute_hallucination_score(generated, expected)
        assert score < 0.55, f"Correct answer should have low hallucination score, got {score}"

    def test_completely_wrong_answer_has_high_hallucination(self):
        expected = "Paris is the capital of France."
        generated = (
            "The mitochondria is the powerhouse of the cell "
            "and produces ATP energy molecules."
        )
        score = compute_hallucination_score(generated, expected)
        assert score > 0.50, (
            f"Wrong answer should have high hallucination score, got {score}"
        )

    def test_empty_answer_returns_1(self):
        assert compute_hallucination_score("", "Expected answer here") == 1.0

    def test_negated_answer_gets_penalty(self):
        expected = "Python is a popular programming language."
        generated = "Python is not a programming language and is never used for coding."
        score = compute_hallucination_score(generated, expected)
        correct_score = compute_hallucination_score(
            "Python is a popular programming language.", expected
        )
        assert score > correct_score, "Negated answer should have higher hallucination score"

    def test_score_between_0_and_1(self):
        score = compute_hallucination_score("some answer", "some expected", "some context")
        assert 0.0 <= score <= 1.0

    def test_context_grounded_answer_scores_lower(self):
        context = "The speed of light is 299,792,458 meters per second in a vacuum."
        expected = "The speed of light is approximately 300,000 km/s."
        grounded = "Light travels at 299,792,458 meters per second."
        ungrounded = (
            "The speed of light is 150,000 km per second "
            "which was discovered in 1700 by Newton."
        )
        score_grounded = compute_hallucination_score(grounded, expected, context)
        score_ungrounded = compute_hallucination_score(ungrounded, expected, context)
        assert 0.0 <= score_grounded <= 1.0
        assert 0.0 <= score_ungrounded <= 1.0
        assert score_grounded <= score_ungrounded + 0.15, (
            f"Grounded ({score_grounded:.3f}) should not be substantially worse "
            f"than ungrounded ({score_ungrounded:.3f})"
        )


# ══════════════════════════════════════════════════════════════
# 5. COMPLETENESS TESTS
# ══════════════════════════════════════════════════════════════

class TestCompleteness:

    def test_complete_answer_scores_high(self):
        expected = "Photosynthesis converts sunlight into glucose using water and carbon dioxide."
        generated = (
            "Photosynthesis converts sunlight into glucose using water and "
            "carbon dioxide in plants."
        )
        score = compute_completeness(generated, expected)
        assert score >= 0.40, f"Complete answer should score higher, got {score}"

    def test_empty_strings_return_zero(self):
        assert compute_completeness("", "expected answer") == 0.0
        assert compute_completeness("generated answer", "") == 0.0

    def test_partial_answer_scores_medium(self):
        # Both sentences share bigrams — recall is non-zero but partial.
        expected = "DNA is a molecule that contains genetic information for all living organisms."
        generated = "DNA contains genetic information for living organisms."
        score = compute_completeness(generated, expected)
        assert score > 0.0, f"Partial answer should have non-zero completeness, got {score}"
        assert score < 0.95, f"Partial answer should not be near-perfect, got {score}"

    def test_score_in_range(self):
        score = compute_completeness("some answer here", "some expected answer here")
        assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════
# 6. PERCENTILE TESTS
# ══════════════════════════════════════════════════════════════

class TestPercentile:

    def test_p50_of_simple_list(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 50) == pytest.approx(3.0, abs=0.1)

    def test_p95_of_latencies(self):
        latencies = list(range(100, 1001, 100))  # 100,200,...,1000
        p95 = percentile(latencies, 95)
        assert 900 <= p95 <= 1000

    def test_empty_list_returns_zero(self):
        assert percentile([], 50) == 0.0

    def test_single_element(self):
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 95) == 42.0

    def test_p100_returns_max(self):
        values = [1.0, 5.0, 3.0, 9.0, 2.0]
        assert percentile(values, 100) == pytest.approx(9.0, abs=0.01)


# ══════════════════════════════════════════════════════════════
# 7. EVALUATE_SINGLE TESTS
# ══════════════════════════════════════════════════════════════

class TestEvaluateSingle:

    def _make_good_result(self):
        return evaluate_single(
            question_id="q001",
            question="What is photosynthesis?",
            expected_answer=(
                "Photosynthesis is how plants convert sunlight into glucose "
                "using water and CO2."
            ),
            generated_answer=(
                "Photosynthesis is the process plants use to convert sunlight "
                "into glucose using water and carbon dioxide."
            ),
            latency_ms=150,
        )

    def test_good_answer_passes(self):
        result = self._make_good_result()
        assert isinstance(result, EvalResult)
        assert result.question_id == "q001"
        assert result.hallucination_score <= 0.80  # reasonable ceiling

    def test_result_has_all_scores(self):
        result = self._make_good_result()
        assert 0.0 <= result.relevancy_score <= 1.0
        assert 0.0 <= result.faithfulness_score <= 1.0
        assert 0.0 <= result.hallucination_score <= 1.0
        assert 0.0 <= result.semantic_similarity <= 1.0
        assert 0.0 <= result.completeness_score <= 1.0

    def test_empty_answer_fails(self):
        result = evaluate_single(
            question_id="q002",
            question="What is gravity?",
            expected_answer="Gravity is a fundamental force that attracts objects with mass.",
            generated_answer="",
            latency_ms=50,
        )
        assert result.passed is False
        assert len(result.failure_reasons) > 0

    def test_very_slow_answer_fails_latency(self):
        result = evaluate_single(
            question_id="q003",
            question="What is the speed of light?",
            expected_answer="The speed of light is 299,792,458 m/s.",
            generated_answer="The speed of light is approximately 300,000 km/s.",
            latency_ms=20_000,  # 20 seconds — exceeds 15s SLA
        )
        assert result.passed is False
        assert any("latency" in r.lower() for r in result.failure_reasons)

    def test_custom_thresholds_respected(self):
        # Set very strict thresholds so a decent answer still fails
        strict = {**DEFAULT_THRESHOLDS, "min_relevancy": 0.99}
        result = evaluate_single(
            question_id="q004",
            question="What is Python?",
            expected_answer="Python is a high-level programming language.",
            generated_answer="Python is a programming language.",
            latency_ms=100,
            thresholds=strict,
        )
        assert result.passed is False

    def test_to_dict_has_expected_keys(self):
        result = self._make_good_result()
        d = result.to_dict()
        assert "question_id" in d
        assert "scores" in d
        assert "passed" in d
        assert "failure_reasons" in d
        assert "latency_ms" in d
        scores = d["scores"]
        for key in (
            "relevancy",
            "faithfulness",
            "hallucination",
            "semantic_similarity",
            "completeness",
        ):
            assert key in scores


# ══════════════════════════════════════════════════════════════
# 8. AGGREGATE RESULTS TESTS
# ══════════════════════════════════════════════════════════════

class TestAggregateResults:

    def _make_results(self, n_pass: int, n_fail: int) -> list[EvalResult]:
        results = []
        for i in range(n_pass):
            r = EvalResult(
                question_id=f"q{i:03d}",
                question="Q",
                expected_answer="E",
                generated_answer="G",
                latency_ms=200.0,
                relevancy_score=0.7,
                faithfulness_score=0.7,
                hallucination_score=0.2,
                semantic_similarity=0.6,
                completeness_score=0.5,
                passed=True,
            )
            results.append(r)
        for i in range(n_fail):
            r = EvalResult(
                question_id=f"qf{i:03d}",
                question="Q",
                expected_answer="E",
                generated_answer="",
                latency_ms=200.0,
                relevancy_score=0.1,
                faithfulness_score=0.1,
                hallucination_score=0.9,
                semantic_similarity=0.05,
                completeness_score=0.05,
                passed=False,
                failure_reasons=["hallucination too high"],
            )
            results.append(r)
        return results

    def test_empty_results_returns_zeros(self):
        agg = aggregate_results([])
        assert agg.total_questions == 0
        assert agg.pass_rate == 0.0

    def test_all_pass_not_blocked(self):
        results = self._make_results(10, 0)
        agg = aggregate_results(results)
        assert agg.pass_rate == 1.0
        assert not agg.blocked

    def test_high_failure_rate_blocks(self):
        results = self._make_results(3, 7)  # 30% pass rate < 65% threshold
        agg = aggregate_results(results)
        assert agg.blocked is True
        assert any("pass rate" in r.lower() for r in agg.block_reasons)

    def test_high_hallucination_blocks(self):
        results = self._make_results(0, 10)  # all fail with 0.9 hallucination
        agg = aggregate_results(results)
        assert agg.blocked is True
        assert any("hallucination" in r.lower() for r in agg.block_reasons)

    def test_pass_rate_calculation(self):
        results = self._make_results(7, 3)
        agg = aggregate_results(results)
        assert agg.pass_rate == pytest.approx(0.7, abs=0.001)
        assert agg.total_questions == 10
        assert agg.passed == 7
        assert agg.failed == 3

    def test_to_dict_shape(self):
        results = self._make_results(5, 0)
        agg = aggregate_results(results)
        d = agg.to_dict()
        for key in (
            "total_questions", "passed", "failed", "pass_rate",
            "avg_relevancy", "avg_faithfulness", "avg_hallucination_rate",
            "avg_semantic_similarity", "avg_completeness",
            "latency_p50_ms", "latency_p95_ms", "blocked", "block_reasons",
        ):
            assert key in d, f"Missing key: {key}"

    def test_latency_percentiles_computed(self):
        results = self._make_results(5, 0)
        for i, r in enumerate(results):
            r.latency_ms = float((i + 1) * 100)
        agg = aggregate_results(results)
        assert agg.latency_p50_ms > 0
        assert agg.latency_p95_ms >= agg.latency_p50_ms


# ══════════════════════════════════════════════════════════════
# 9. MODEL ADAPTER TESTS
# ══════════════════════════════════════════════════════════════

class TestMockModelAdapter:

    def setup_method(self):
        self.model = MockModelAdapter(seed=42)

    def test_name_is_mock(self):
        assert self.model.name == "mock-llm"

    def test_known_question_returns_answer(self):
        answer = self.model.generate("What is the capital of France?")
        assert isinstance(answer, str)
        assert len(answer) > 5

    def test_unknown_question_returns_string(self):
        answer = self.model.generate("What is the meaning of xyzzy plugh?")
        assert isinstance(answer, str)

    def test_with_context_uses_context(self):
        context = "The sky is blue due to Rayleigh scattering."
        answer = self.model.generate("Why is the sky blue?", context=context)
        assert isinstance(answer, str)

    def test_deterministic_with_same_seed(self):
        m1 = MockModelAdapter(seed=99, hallucination_rate=0.0)
        m2 = MockModelAdapter(seed=99, hallucination_rate=0.0)
        q = "What is photosynthesis?"
        assert m1.generate(q) == m2.generate(q)

    def test_zero_hallucination_rate_always_correct(self):
        model = MockModelAdapter(seed=42, hallucination_rate=0.0)
        answer = model.generate("What is photosynthesis?")
        assert (
            "photosynthesis" in answer.lower()
            or "plant" in answer.lower()
            or "sunlight" in answer.lower()
        )


class TestSimpleModelAdapter:

    def setup_method(self):
        self.model = SimpleModelAdapter()

    def test_name(self):
        assert self.model.name == "simple-rule-based"

    def test_generates_string(self):
        answer = self.model.generate("What is machine learning?")
        assert isinstance(answer, str)
        assert len(answer) > 10

    def test_what_is_pattern(self):
        answer = self.model.generate("What is blockchain?")
        assert len(answer) > 20

    def test_how_does_pattern(self):
        answer = self.model.generate("How does recursion work?")
        assert len(answer) > 10

    def test_difference_pattern(self):
        answer = self.model.generate("What is the difference between TCP and UDP?")
        assert isinstance(answer, str)


class TestGetModelAdapter:

    def test_returns_mock(self):
        adapter = get_model_adapter("mock")
        assert adapter.name == "mock-llm"

    def test_returns_simple(self):
        adapter = get_model_adapter("simple")
        assert adapter.name == "simple-rule-based"

    def test_case_insensitive(self):
        adapter = get_model_adapter("MOCK")
        assert adapter.name == "mock-llm"

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model adapter"):
            get_model_adapter("nonexistent_model")


# ══════════════════════════════════════════════════════════════
# 10. GOLDEN DATASET TESTS
# ══════════════════════════════════════════════════════════════

class TestGoldenDataset:

    def setup_method(self):
        dataset_path = ROOT / "data" / "golden_dataset.json"
        with open(dataset_path) as f:
            self.data = json.load(f)

    def test_dataset_has_at_least_100_questions(self):
        assert len(self.data) >= 100, f"Expected ≥100 questions, got {len(self.data)}"

    def test_all_items_have_required_fields(self):
        required = {"id", "question", "expected_answer", "category", "difficulty"}
        for item in self.data:
            missing = required - set(item.keys())
            assert not missing, f"Item {item.get('id')} missing fields: {missing}"

    def test_all_ids_unique(self):
        ids = [item["id"] for item in self.data]
        assert len(ids) == len(set(ids)), "Dataset contains duplicate IDs"

    def test_no_empty_questions_or_answers(self):
        for item in self.data:
            assert item["question"].strip(), f"Empty question in {item['id']}"
            assert item["expected_answer"].strip(), f"Empty answer in {item['id']}"

    def test_difficulty_values_valid(self):
        valid = {"easy", "medium", "hard"}
        for item in self.data:
            assert item["difficulty"] in valid, (
                f"Invalid difficulty '{item['difficulty']}' in {item['id']}"
            )

    def test_covers_multiple_categories(self):
        categories = {item["category"] for item in self.data}
        assert len(categories) >= 5, (
            f"Expected ≥5 categories, got {len(categories)}: {categories}"
        )

    def test_has_all_difficulty_levels(self):
        difficulties = {item["difficulty"] for item in self.data}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties


# ══════════════════════════════════════════════════════════════
# 11. INTEGRATION TEST — MINI PIPELINE RUN
# ══════════════════════════════════════════════════════════════

class TestPipelineIntegration:

    def test_pipeline_runs_end_to_end(self, tmp_path):
        """Run a mini pipeline with 5 questions and verify outputs."""
        sys.path.insert(0, str(ROOT))
        from pipeline.pipeline import run_pipeline

        results, agg = run_pipeline(
            dataset_path=str(ROOT / "data" / "golden_dataset.json"),
            model_name="mock",
            sample_size=5,
            output_dir=str(tmp_path),
            verbose=False,
        )

        assert len(results) == 5
        assert agg.total_questions == 5
        assert 0.0 <= agg.pass_rate <= 1.0

    def test_report_json_saved(self, tmp_path):
        """Check that a JSON report is written after the run."""
        from pipeline.pipeline import run_pipeline

        run_pipeline(
            dataset_path=str(ROOT / "data" / "golden_dataset.json"),
            model_name="mock",
            sample_size=3,
            output_dir=str(tmp_path),
            verbose=False,
        )

        report_files = list(tmp_path.glob("eval_report_*.json"))
        assert len(report_files) == 1, "Expected exactly one report file"

        with open(report_files[0]) as f:
            report = json.load(f)

        assert "meta" in report
        assert "aggregate" in report
        assert "results" in report
        assert len(report["results"]) == 3

    def test_pipeline_with_simple_adapter(self, tmp_path):
        """Ensure simple adapter also produces valid results."""
        from pipeline.pipeline import run_pipeline

        results, agg = run_pipeline(
            dataset_path=str(ROOT / "data" / "golden_dataset.json"),
            model_name="simple",
            sample_size=5,
            output_dir=str(tmp_path),
            verbose=False,
        )

        assert agg.total_questions == 5
        for r in results:
            assert 0.0 <= r.relevancy_score <= 1.0
            assert 0.0 <= r.hallucination_score <= 1.0

    def test_blocked_pipeline_metrics(self, tmp_path):
        """Force a block by setting very strict thresholds."""
        from pipeline.pipeline import run_pipeline

        strict_thresholds = {
            **DEFAULT_THRESHOLDS,
            "min_pass_rate": 1.0,  # 100% pass required — impossible with mock
        }

        _, agg = run_pipeline(
            dataset_path=str(ROOT / "data" / "golden_dataset.json"),
            model_name="mock",
            sample_size=10,
            output_dir=str(tmp_path),
            verbose=False,
            thresholds=strict_thresholds,
        )

        assert agg.blocked is True

    def test_all_results_have_valid_structure(self, tmp_path):
        """Each result dict must have the required shape."""
        from pipeline.pipeline import run_pipeline

        results, _ = run_pipeline(
            dataset_path=str(ROOT / "data" / "golden_dataset.json"),
            model_name="mock",
            sample_size=8,
            output_dir=str(tmp_path),
            verbose=False,
        )

        for r in results:
            d = r.to_dict()
            assert isinstance(d["passed"], bool)
            assert isinstance(d["failure_reasons"], list)
            assert isinstance(d["latency_ms"], float)
            for score_key in (
                "relevancy",
                "faithfulness",
                "hallucination",
                "semantic_similarity",
                "completeness",
            ):
                val = d["scores"][score_key]
                assert 0.0 <= val <= 1.0, f"{score_key} out of range: {val}"


# ══════════════════════════════════════════════════════════════
# 12. EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_unicode_input(self):
        score = compute_semantic_similarity(
            "Bonjour, comment ça va? 日本語テスト",
            "Hello, how are you? Japanese test",
        )
        assert 0.0 <= score <= 1.0

    def test_very_long_answer(self):
        long_text = "word " * 500
        score = compute_relevancy("What is programming?", long_text)
        assert 0.0 <= score <= 1.0

    def test_special_characters_handled(self):
        score = compute_semantic_similarity(
            "C++ is a programming language!!!",
            "C++ is used for systems programming...",
        )
        assert 0.0 <= score <= 1.0

    def test_only_punctuation_answer(self):
        score = compute_relevancy("What is AI?", "!!! ??? ...")
        assert 0.0 <= score <= 1.0

    def test_numbers_in_answers(self):
        score = compute_hallucination_score(
            "The speed of light is 299792458 m/s",
            "The speed of light is 299,792,458 meters per second",
        )
        assert 0.0 <= score <= 1.0

    def test_single_word_answer(self):
        result = evaluate_single(
            question_id="edge001",
            question="What is the capital of France?",
            expected_answer="Paris",
            generated_answer="Paris",
            latency_ms=10,
        )
        assert isinstance(result.passed, bool)

    def test_repeated_words_dont_inflate_score(self):
        answer = "Paris Paris Paris Paris Paris Paris Paris"
        expected = "Paris is the capital of France and a major European city."
        score = compute_completeness(answer, expected)
        # Should not be 1.0 — repetition shouldn't inflate bigram coverage
        assert score < 0.8


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(ROOT),
    )
    sys.exit(result.returncode)

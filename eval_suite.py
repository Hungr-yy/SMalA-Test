"""
eval_suite.py
=============
Measures student SLM performance on malware analysis benchmarks.

Metrics
-------
* **Accuracy** – fraction of questions where the model's selected options exactly
  match the gold answer set (strict).
* **Average Jaccard Score** – partial-credit metric: |prediction ∩ gold| / |prediction ∪ gold|.
  This allows SLMs that are partially correct to receive credit, and is the
  primary "soft" metric used in the CyberSOCEval malware analysis benchmark.

The module is compatible with the question format used in CyberSOCEval and can
be used both for standalone evaluation and within the iterative distillation loop.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def jaccard_score(pred: set[str], gold: set[str]) -> float:
    """Return the Jaccard similarity between two sets of option labels.

    Parameters
    ----------
    pred:
        The set of option labels chosen by the model (e.g., ``{"A", "C"}``).
    gold:
        The set of correct option labels.

    Returns
    -------
    float
        Value in [0, 1].  Returns 1.0 when both sets are empty.
    """
    if not pred and not gold:
        return 1.0
    intersection = len(pred & gold)
    union = len(pred | gold)
    return intersection / union if union > 0 else 0.0


def exact_match(pred: set[str], gold: set[str]) -> bool:
    """Return ``True`` if *pred* exactly equals *gold*."""
    return pred == gold


def _parse_answer(raw: str) -> set[str]:
    """Extract option labels (A–J) from a free-form model response string."""
    # Match standalone capital letters or comma/space-separated sequences
    labels = re.findall(r"\b([A-J])\b", raw.upper())
    return set(labels)


# ---------------------------------------------------------------------------
# EvalSuite
# ---------------------------------------------------------------------------

class EvalSuite:
    """Evaluate an SLM on a set of malware analysis questions.

    Parameters
    ----------
    questions_path:
        Path to a JSON file containing a list of question dicts.
        Each dict must have ``"question"``, ``"options"``, and ``"answer"`` keys.
    model_fn:
        A callable that takes a prompt string and returns the model's raw
        text response.  In the distillation loop this is usually
        ``student_trainer.StudentTrainer.generate``.
    truncate_input:
        When ``True``, apply :class:`data_filter.DataFilter` to any ``context``
        field in the question before building the prompt.
    """

    def __init__(
        self,
        questions_path: str | Path,
        model_fn,
        truncate_input: bool = False,
    ) -> None:
        self.questions_path = Path(questions_path)
        self.model_fn = model_fn
        self.truncate_input = truncate_input
        self._questions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load questions from :attr:`questions_path`."""
        with open(self.questions_path) as fh:
            self._questions = json.load(fh)
        logger.info("Loaded %d questions from %s", len(self._questions), self.questions_path)

    # ------------------------------------------------------------------
    # Running evaluation
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the full evaluation and return a stats dict.

        Returns
        -------
        dict
            Keys:
            - ``accuracy``        : float – exact-match accuracy
            - ``avg_jaccard``     : float – average Jaccard score
            - ``num_questions``   : int
            - ``results``         : list of per-question dicts
            - ``by_difficulty``   : dict mapping difficulty label → accuracy
            - ``by_topic``        : dict mapping topic → accuracy
            - ``by_malware_type`` : dict mapping malware family → accuracy
        """
        if not self._questions:
            self.load()

        results = []
        for q in self._questions:
            prompt = self._build_prompt(q)
            raw_response = self.model_fn(prompt)
            pred = _parse_answer(raw_response)
            gold = set(q.get("answer", []))

            em = exact_match(pred, gold)
            jac = jaccard_score(pred, gold)

            results.append(
                {
                    "id": q.get("id", ""),
                    "question": q.get("question", ""),
                    "gold": sorted(gold),
                    "pred": sorted(pred),
                    "exact_match": em,
                    "jaccard": jac,
                    "difficulty": q.get("difficulty", "unknown"),
                    "topic": q.get("topic", "unknown"),
                    "malware_type": q.get("malware_type", "unknown"),
                }
            )

        stats = self._aggregate(results)
        stats["results"] = results
        return stats

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, q: dict[str, Any]) -> str:
        """Build the evaluation prompt for a single question."""
        lines = []

        context = q.get("context", "")
        if context and self.truncate_input:
            try:
                from data_filter import DataFilter
                context = DataFilter(mode="essential").filter_string(context)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass  # context may be plain text rather than JSON

        if context:
            lines.append("=== Malware Detonation Report ===")
            lines.append(str(context))
            lines.append("")

        lines.append(q.get("question", ""))
        lines.append("")

        options: dict[str, str] = q.get("options", {})
        for label, text in sorted(options.items()):
            lines.append(f"{label}. {text}")

        lines.append("")
        lines.append(
            "Select ALL correct options. Reply with only the option letters "
            "separated by commas (e.g., 'A, C')."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate metrics from per-question results."""
        if not results:
            return {
                "accuracy": 0.0,
                "avg_jaccard": 0.0,
                "num_questions": 0,
                "by_difficulty": {},
                "by_topic": {},
                "by_malware_type": {},
            }

        accuracy = sum(r["exact_match"] for r in results) / len(results)
        avg_jaccard = sum(r["jaccard"] for r in results) / len(results)

        def _breakdown(key: str) -> dict[str, float]:
            groups: dict[str, list[float]] = {}
            for r in results:
                cat = r.get(key, "unknown")
                groups.setdefault(cat, []).append(float(r["exact_match"]))
            return {cat: sum(vals) / len(vals) for cat, vals in groups.items()}

        return {
            "accuracy": round(accuracy, 4),
            "avg_jaccard": round(avg_jaccard, 4),
            "num_questions": len(results),
            "by_difficulty": _breakdown("difficulty"),
            "by_topic": _breakdown("topic"),
            "by_malware_type": _breakdown("malware_type"),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, path: str | Path, stats: dict[str, Any]) -> None:
        """Write evaluation stats to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(stats, fh, indent=2)
        logger.info("Results written to %s", path)

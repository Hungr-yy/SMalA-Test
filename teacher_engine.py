"""
teacher_engine.py
=================
Interacts with a high-capability "teacher" LLM (OpenAI GPT-4o, Google Gemini,
or Alibaba Qwen) to drive the three core steps of the distillation loop:

  1. Exam Generation   – create multiple-choice questions from a malware report
  2. Evaluation        – score the student's answers and identify weaknesses
  3. Curriculum Generation – produce synthetic training examples targeting weaknesses
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
SUPPORTED_PROVIDERS = ("openai", "google", "qwen", "together")


def _build_openai_client(api_key: str):
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key)


def _build_google_client(api_key: str):
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    return genai


def _build_together_client(api_key: str):
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")


# ---------------------------------------------------------------------------
# TeacherEngine
# ---------------------------------------------------------------------------

class TeacherEngine:
    """Wraps a teacher LLM and exposes exam / evaluation / curriculum methods.

    Parameters
    ----------
    provider:
        One of ``"openai"``, ``"google"``, ``"qwen"``, or ``"together"``.
    model:
        Model identifier, e.g. ``"gpt-4o"``, ``"gemini-2.5-pro"``, ``"qwen-max"``.
    api_key:
        API key for the chosen provider.  Defaults to the corresponding
        ``*_API_KEY`` environment variable when not supplied.
    temperature:
        Sampling temperature used for all completions.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        provider = provider.lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from: {SUPPORTED_PROVIDERS}"
            )
        self.provider = provider
        self.model = model
        self.temperature = temperature

        if api_key is None:
            env_map = {
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",
                "qwen": "QWEN_API_KEY",
                "together": "TOGETHER_API_KEY",
            }
            api_key = os.environ.get(env_map[provider], "")

        if provider == "openai":
            self._client = _build_openai_client(api_key)
        elif provider == "google":
            self._client = _build_google_client(api_key)
            self._google_model = self._client.GenerativeModel(model)
        elif provider in ("qwen", "together"):
            self._client = _build_together_client(api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # ------------------------------------------------------------------
    # Internal completion helper
    # ------------------------------------------------------------------

    def _complete(self, prompt: str) -> str:
        """Return a text completion for *prompt* using the teacher LLM."""
        if self.provider == "google":
            response = self._google_model.generate_content(prompt)
            return response.text

        # OpenAI-compatible path (openai / together / qwen)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Step 1 – Exam Generation
    # ------------------------------------------------------------------

    def generate_exam(
        self,
        report_summary: str,
        num_questions: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate multiple-choice exam questions from a malware report.

        Parameters
        ----------
        report_summary:
            A condensed string representation of the malware detonation report
            (e.g., produced by :class:`report_parser.ReportParser`).
        num_questions:
            How many questions to generate.

        Returns
        -------
        list[dict]
            A list of question dicts, each with keys:
            ``question``, ``options`` (list), ``answer`` (list of correct option labels).
        """
        prompt = textwrap.dedent(f"""
            You are a malware analysis expert creating an exam for a junior analyst.

            Below is a summary of a malware sandbox detonation report:
            ---
            {report_summary}
            ---

            Generate {num_questions} multiple-choice questions to test understanding
            of the malware's behavior, attack techniques, and indicators of compromise.

            Rules:
            - Each question must have between 4 and 10 answer options labelled A–J.
            - One or more options may be correct (multi-select).
            - Return a JSON array where each element has the keys:
              "question" (string), "options" (object mapping label→text),
              "answer" (array of correct labels).
            - Return ONLY valid JSON, no prose.
        """).strip()

        raw = self._complete(prompt)
        return self._parse_json(raw, default=[])

    # ------------------------------------------------------------------
    # Step 3 – Evaluation & Feedback
    # ------------------------------------------------------------------

    def evaluate_exam(
        self,
        questions: list[dict[str, Any]],
        student_answers: list[list[str]],
    ) -> dict[str, Any]:
        """Score the student's answers and identify conceptual weaknesses.

        Parameters
        ----------
        questions:
            The exam questions as returned by :meth:`generate_exam`.
        student_answers:
            A list of answer-label lists, one per question.

        Returns
        -------
        dict
            Keys: ``score`` (float 0–1), ``feedback`` (str),
            ``weaknesses`` (list of topic strings).
        """
        qa_pairs = []
        for i, (q, ans) in enumerate(zip(questions, student_answers)):
            qa_pairs.append(
                f"Q{i+1}: {q['question']}\n"
                f"  Correct: {q.get('answer', [])}\n"
                f"  Student: {ans}"
            )

        prompt = textwrap.dedent(f"""
            You are evaluating a student's performance on a malware analysis exam.

            Questions and answers:
            {chr(10).join(qa_pairs)}

            Produce a JSON object with:
            - "score": float between 0 and 1 (fraction of fully correct answers)
            - "feedback": a paragraph explaining overall performance
            - "weaknesses": a list of specific topic strings the student struggled with
              (e.g., "process injection detection", "MITRE ATT&CK mapping")

            Return ONLY valid JSON.
        """).strip()

        raw = self._complete(prompt)
        return self._parse_json(raw, default={"score": 0.0, "feedback": "", "weaknesses": []})

    # ------------------------------------------------------------------
    # Step 4 – Curriculum Generation
    # ------------------------------------------------------------------

    def generate_curriculum(
        self,
        weaknesses: list[str],
        num_examples: int = 100,
    ) -> list[dict[str, Any]]:
        """Produce a synthetic training dataset targeting the student's weaknesses.

        Parameters
        ----------
        weaknesses:
            Topics identified by :meth:`evaluate_exam`.
        num_examples:
            Number of training examples to generate (10 k–100 k in production;
            keep small for quick iteration).

        Returns
        -------
        list[dict]
            Training examples, each with keys ``instruction``, ``input``, ``output``.
        """
        topics = ", ".join(weaknesses) if weaknesses else "general malware analysis"
        prompt = textwrap.dedent(f"""
            You are building a training curriculum for a small language model that
            specialises in malware reverse engineering.

            The student struggles with: {topics}

            Generate {num_examples} diverse training examples that target these weaknesses.
            Each example should be a realistic malware analysis task.

            Return a JSON array where each element has:
            - "instruction": the task description
            - "input": relevant context (e.g., a snippet of a sandbox report or assembly)
            - "output": the ideal expert response

            Return ONLY valid JSON.
        """).strip()

        raw = self._complete(prompt)
        return self._parse_json(raw, default=[])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str, default: Any) -> Any:
        """Attempt to extract and parse JSON from a raw LLM response."""
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON; returning default.")
            logger.debug("Raw response: %s", raw)
            return default

"""
main.py
=======
Orchestrates the iterative teacher-student knowledge distillation loop:

  1. Generate an exam from a malware detonation report  (teacher)
  2. Student attempts the exam
  3. Teacher evaluates the student's answers
  4. Teacher generates a targeted curriculum
  5. Fine-tune the student with LoRA/QLoRA
  6. Repeat until the target accuracy is reached or max rounds exhausted

Usage
-----
    python main.py --config configs/model_config.yaml [options]

    Options:
      --config PATH            Path to model_config.yaml (default: configs/model_config.yaml)
      --report PATH            Path to a malware detonation report JSON to use as exam source
      --rounds INT             Maximum distillation rounds (default: 5)
      --target-accuracy FLOAT  Stop when student accuracy exceeds this threshold (default: 0.80)
      --output-dir PATH        Where to save adapters and results (default: outputs/)
      --eval-questions PATH    Path to benchmark questions JSON for evaluation
      --truncate-input         Apply DataFilter truncation during evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Distillation loop
# ---------------------------------------------------------------------------

def run_distillation_loop(
    config: dict[str, Any],
    report_path: str | None,
    rounds: int,
    target_accuracy: float,
    output_dir: str,
    eval_questions_path: str | None,
    truncate_input: bool,
) -> None:
    from teacher_engine import TeacherEngine
    from student_trainer import StudentTrainer
    from report_parser import ReportParser
    from eval_suite import EvalSuite

    # ------------------------------------------------------------------
    # Initialise teacher
    # ------------------------------------------------------------------
    teacher_cfg = config.get("teacher", {})
    teacher = TeacherEngine(
        provider=teacher_cfg.get("provider", "openai"),
        model=teacher_cfg.get("model", "gpt-4o"),
        api_key=teacher_cfg.get("api_key") or None,
        temperature=teacher_cfg.get("temperature", 0.7),
    )
    logger.info(
        "Teacher: %s / %s", teacher_cfg.get("provider"), teacher_cfg.get("model")
    )

    # ------------------------------------------------------------------
    # Initialise student
    # ------------------------------------------------------------------
    student = StudentTrainer.from_config(config["_config_path"])
    logger.info("Student: %s", config.get("student", {}).get("model_name_or_path"))

    # ------------------------------------------------------------------
    # Load report
    # ------------------------------------------------------------------
    report_summary = ""
    if report_path:
        parser = ReportParser(truncate=truncate_input)
        parsed = parser.parse_file(report_path)
        report_summary = parser.to_summary_string(parsed)
        logger.info("Loaded report from %s", report_path)
    else:
        logger.warning(
            "No --report provided; using empty report summary for exam generation."
        )

    # ------------------------------------------------------------------
    # Iterative loop
    # ------------------------------------------------------------------
    best_accuracy = 0.0

    for round_num in range(1, rounds + 1):
        logger.info("=" * 60)
        logger.info("Distillation round %d / %d", round_num, rounds)
        logger.info("=" * 60)

        # Step 1 – Generate exam
        logger.info("[Step 1] Generating exam …")
        num_q = config.get("exam", {}).get("num_questions", 5)
        questions = teacher.generate_exam(report_summary, num_questions=num_q)
        if not questions:
            logger.error("Teacher returned no questions; skipping round.")
            continue

        exam_path = Path(output_dir) / f"round_{round_num}_exam.json"
        exam_path.parent.mkdir(parents=True, exist_ok=True)
        with open(exam_path, "w") as fh:
            json.dump(questions, fh, indent=2)
        logger.info("Exam saved to %s", exam_path)

        # Step 2 – Student attempts exam
        logger.info("[Step 2] Student attempting exam …")
        student_answers: list[list[str]] = []
        for q in questions:
            prompt = _build_exam_prompt(q)
            raw = student.generate(prompt, max_new_tokens=64)
            labels = re.findall(r"\b([A-J])\b", raw.upper())
            student_answers.append(labels)
            logger.debug("Q: %s  →  %s", q.get("question", "")[:60], labels)

        # Step 3 – Evaluate
        logger.info("[Step 3] Teacher evaluating answers …")
        evaluation = teacher.evaluate_exam(questions, student_answers)
        score = evaluation.get("score", 0.0)
        weaknesses = evaluation.get("weaknesses", [])
        logger.info(
            "Evaluation score: %.2f  |  Weaknesses: %s", score, weaknesses
        )

        eval_path = Path(output_dir) / f"round_{round_num}_eval.json"
        with open(eval_path, "w") as fh:
            json.dump(evaluation, fh, indent=2)

        # Step 4 – Generate curriculum
        logger.info("[Step 4] Generating curriculum …")
        num_examples = config.get("curriculum", {}).get("num_examples", 100)
        curriculum = teacher.generate_curriculum(weaknesses, num_examples=num_examples)
        logger.info("Curriculum: %d examples", len(curriculum))

        curriculum_path = Path(output_dir) / f"round_{round_num}_curriculum.json"
        with open(curriculum_path, "w") as fh:
            json.dump(curriculum, fh, indent=2)

        if not curriculum:
            logger.warning("Empty curriculum; skipping fine-tuning this round.")
            continue

        # Step 5 – Fine-tune student
        logger.info("[Step 5] Fine-tuning student …")
        adapter_path = str(Path(output_dir) / f"round_{round_num}_adapter")
        student.output_dir = adapter_path
        student.train(curriculum)
        student.save(adapter_path)

        # Step 6 – Optional benchmark evaluation
        if eval_questions_path:
            logger.info("[Step 6] Running benchmark evaluation …")
            suite = EvalSuite(
                questions_path=eval_questions_path,
                model_fn=student.generate,
                truncate_input=truncate_input,
            )
            stats = suite.run()
            accuracy = stats["accuracy"]
            logger.info(
                "Benchmark accuracy: %.4f  |  Avg Jaccard: %.4f",
                accuracy,
                stats["avg_jaccard"],
            )
            suite.save_results(
                Path(output_dir) / f"round_{round_num}_stats.json", stats
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Symlink / copy best adapter
                best_path = Path(output_dir) / "best_adapter"
                if best_path.exists() or best_path.is_symlink():
                    best_path.unlink()
                best_path.symlink_to(Path(adapter_path).resolve())
                logger.info("New best accuracy %.4f; adapter at %s", best_accuracy, best_path)

            if accuracy >= target_accuracy:
                logger.info(
                    "Target accuracy %.2f reached (%.4f). Stopping.",
                    target_accuracy,
                    accuracy,
                )
                break
        else:
            logger.info("No --eval-questions provided; skipping benchmark step.")

    logger.info("Distillation loop complete. Best accuracy: %.4f", best_accuracy)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_exam_prompt(q: dict[str, Any]) -> str:
    lines = [q.get("question", ""), ""]
    options: dict[str, str] = q.get("options", {})
    for label, text in sorted(options.items()):
        lines.append(f"{label}. {text}")
    lines.append("")
    lines.append(
        "Select ALL correct options. Reply with only the option letters separated by commas."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SMalRe – iterative teacher-student distillation for malware analysis"
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to model_config.yaml",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to a malware detonation report JSON file",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Maximum number of distillation rounds",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.80,
        dest="target_accuracy",
        help="Stop early when benchmark accuracy exceeds this value",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        dest="output_dir",
        help="Directory for adapters, curricula, and results",
    )
    parser.add_argument(
        "--eval-questions",
        default=None,
        dest="eval_questions",
        help="Path to benchmark questions JSON (CyberSOCEval format)",
    )
    parser.add_argument(
        "--truncate-input",
        action="store_true",
        dest="truncate_input",
        help="Apply DataFilter truncation to report context before evaluation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    config = load_config(args.config)
    config["_config_path"] = args.config  # pass path for StudentTrainer.from_config

    run_distillation_loop(
        config=config,
        report_path=args.report,
        rounds=args.rounds,
        target_accuracy=args.target_accuracy,
        output_dir=args.output_dir,
        eval_questions_path=args.eval_questions,
        truncate_input=args.truncate_input,
    )


if __name__ == "__main__":
    main()

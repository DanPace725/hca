import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hybrid_reasoner import Observation, build_default_controller
from hybrid_reasoner.config.settings import RuntimeConfig


def normalize_answer(text: str) -> str:
    """Simple normalization for string comparison."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def flatten_json_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        return " ".join(flatten_json_payload(v) for v in payload.values())
    if isinstance(payload, list):
        return " ".join(flatten_json_payload(v) for v in payload)
    return str(payload)


def normalize_predicted(predicted: str, output_shape: str) -> Tuple[str, bool]:
    if output_shape == "json":
        try:
            data = json.loads(predicted)
            return normalize_answer(flatten_json_payload(data)), True
        except Exception:
            return normalize_answer(predicted), False
    return normalize_answer(predicted), True


def numbers_in(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def compute_match(pred_norm: str, exp_norm: str) -> Tuple[bool, str]:
    if pred_norm == exp_norm:
        return True, "exact"
    if exp_norm and exp_norm in pred_norm:
        return True, "expected_in_predicted"
    if pred_norm and pred_norm in exp_norm:
        return True, "predicted_in_expected"
    pred_nums = numbers_in(pred_norm)
    exp_nums = numbers_in(exp_norm)
    if pred_nums and exp_nums:
        if pred_nums == exp_nums:
            return True, "numbers_match"
        if sorted(pred_nums) == sorted(exp_nums):
            return True, "numbers_set_match"
        if set(exp_nums).issubset(set(pred_nums)):
            return True, "numbers_subset"
    return False, "no_match"


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Task file must contain a list of tasks.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid reasoner on a task file.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="dataset/test_prompts.json",
        help="Path to JSON task file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of tasks to run.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Filter tasks by scenario (can be repeated or comma-separated).",
    )
    parser.add_argument(
        "--log-traces",
        action="store_true",
        help="Enable per-episode trace logging to the runs directory.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=RuntimeConfig.max_steps,
        help=f"Maximum reasoning steps (default {RuntimeConfig.max_steps}).",
    )
    parser.add_argument(
        "--halt-threshold",
        type=float,
        default=RuntimeConfig.halt_threshold,
        help=f"Halting threshold (default {RuntimeConfig.halt_threshold}).",
    )
    parser.add_argument(
        "--core-checkpoint",
        type=str,
        default=None,
        help="Path to a GRU core checkpoint to load (optional).",
    )
    parser.add_argument(
        "--use-rules",
        action="store_true",
        help="Enable rule-based shortcuts (disabled by default - use for comparison testing only).",
    )
    parser.add_argument(
        "--semantic-fallback",
        action="store_true",
        help="Allow semantic engine to fallback to stub on parse failure (useful when LLM JSON is flaky).",
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    tasks = load_tasks(tasks_path)
    scenarios = set()
    if args.scenario:
        for item in args.scenario:
            scenarios.update(s.strip() for s in item.split(",") if s.strip())
        if scenarios:
            tasks = [t for t in tasks if t.get("scenario") in scenarios]
    if args.limit:
        tasks = tasks[: args.limit]

    runtime_config = RuntimeConfig(
        max_steps=args.max_steps,
        halt_threshold=args.halt_threshold,
        log_traces=args.log_traces,
    )
    from hybrid_reasoner.config.reasoning import ReasoningConfig
    from hybrid_reasoner.config.settings import ModelConfig
    from hybrid_reasoner.semantic.engine import SemanticEngine
    reasoning_config = ReasoningConfig(
        use_rules=args.use_rules,  # Rules disabled by default
        use_gru=True,
        max_steps=runtime_config.max_steps,
        halt_threshold=runtime_config.halt_threshold,
    )
    controller = build_default_controller(runtime_config)
    controller.reasoning_config = reasoning_config
    controller.reasoning_core.config = reasoning_config
    if args.semantic_fallback:
        controller.semantic_engine = SemanticEngine(ModelConfig(fallback_to_stub=True))
    
    if args.core_checkpoint:
        try:
            controller.reasoning_core.load_checkpoint(args.core_checkpoint)
            controller.reasoning_config.use_gru = True
            controller.reasoning_core.config.use_gru = True
        except Exception as exc:
            print(f"Warning: failed to load checkpoint: {exc}")

    results = []
    for task in tasks:
        prompt = task.get("prompt", "")
        expected = task.get("expected_answer", "")
        features = task.get("features") or {}
        output_shape = features.get("output_shape", "plain_prose")

        observation = Observation(
            text=prompt,
            metadata={
                "scenario": task.get("scenario"),
                "test_id": task.get("test_id"),
                "output_shape": output_shape,
            },
        )
        predicted, trace = controller.run(observation)

        normalized_predicted, format_ok = normalize_predicted(predicted, output_shape)
        normalized_expected = normalize_answer(expected)
        match, reason = compute_match(normalized_predicted, normalized_expected)
        if not format_ok:
            match = False
            reason = "format_error"

        # Determine what produced the answer
        answer_source = "unknown"
        if trace.steps:
            core_used = trace.steps[0].notes.get("core", "unknown")
            if core_used == "reasoning_core_rule":
                answer_source = "RULES"
            elif core_used == "reasoning_core_gru":
                answer_source = "GRU"
            else:
                answer_source = "STUB"
        
        results.append(
            {
                "test_id": task.get("test_id"),
                "scenario": task.get("scenario"),
                "output_shape": output_shape,
                "expected": expected,
                "predicted": predicted,
                "answer_source": answer_source,  # NEW: Show what produced this answer
                "match": match,
                "reason": reason,
                "format_ok": format_ok,
                "steps": len(trace.steps),
                "semantic_source": trace.semantic_info.get("semantic_source"),
                "semantic_error": trace.semantic_info.get("semantic_error"),
            }
        )

    accuracy = sum(1 for r in results if r["match"]) / len(results) if results else 0.0
    semantic_counts = {}
    answer_source_counts = {}
    for r in results:
        src = r.get("semantic_source") or "unknown"
        semantic_counts[src] = semantic_counts.get(src, 0) + 1
        
        ans_src = r.get("answer_source") or "unknown"
        answer_source_counts[ans_src] = answer_source_counts.get(ans_src, 0) + 1

    summary = {
        "task_count": len(results),
        "accuracy": accuracy,
        "max_steps": args.max_steps,
        "halt_threshold": args.halt_threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "semantic_counts": semantic_counts,
        "answer_source_counts": answer_source_counts,  # NEW: Show what produced answers
        "scenarios_filtered": sorted(scenarios) if scenarios else None,
    }

    eval_dir = runtime_config.log_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nRan {len(results)} tasks. Accuracy={accuracy:.2%}")
    print(f"Answer sources: {answer_source_counts}")
    print(f"Semantic sources: {semantic_counts}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

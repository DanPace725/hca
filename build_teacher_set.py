"""
Build a small teacher dataset for GRU training by using the rule-based solvers as teachers.
Outputs JSONL with fields:
  prompt, scenario, output_shape, teacher_answer, has_numbers, token_estimate
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from hybrid_reasoner import Observation, ReasoningConfig
from hybrid_reasoner.core.reasoning_core import ReasoningCore


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Task file must contain a list of tasks.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build teacher dataset using rule-based answers.")
    parser.add_argument("--tasks", type=str, default="dataset/test_prompts.json", help="Input tasks JSON.")
    parser.add_argument("--out", type=str, default="runs/data/teacher.jsonl", help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on tasks.")
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]

    core = ReasoningCore(config=ReasoningConfig(use_rules=True, use_gru=False))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            prompt = task.get("prompt", "")
            scenario = task.get("scenario")
            output_shape = (task.get("features") or {}).get("output_shape") or "plain_prose"
            hint = {
                "scenario": scenario,
                "output_shape": output_shape,
                "observation_text": prompt,
            }
            answer = core._try_solve(hint)  # type: ignore
            if not answer:
                continue
            record = {
                "prompt": prompt,
                "scenario": scenario,
                "output_shape": output_shape,
                "teacher_answer": answer,
                "has_numbers": any(ch.isdigit() for ch in prompt),
                "token_estimate": len(prompt.split()),
            }
            f.write(json.dumps(record) + "\n")
            kept += 1

    print(f"Wrote {kept} teacher examples to {out_path}")


if __name__ == "__main__":
    main()

"""
Lightweight training script for the GRU-based ReasoningCore.
Uses a small subset of dataset/test_prompts.json to provide supervised signals.
If torch is not installed, this script exits gracefully.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None

from hybrid_reasoner import Observation, ReasoningConfig
from hybrid_reasoner.core.reasoning_core import ReasoningCore


def load_json_tasks(path: Path, limit: int) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Task file must contain a list.")
    if limit:
        data = data[:limit]
    return data


def load_teacher_jsonl(path: Path, limit: int) -> List[Tuple[Dict[str, Any], str]]:
    if not path.exists():
        return []
    teacher: List[Tuple[Dict[str, Any], str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            hint = {
                "scenario": obj.get("scenario"),
                "output_shape": obj.get("output_shape"),
                "observation_text": obj.get("prompt", ""),
                "has_numbers": obj.get("has_numbers", any(ch.isdigit() for ch in obj.get("prompt", ""))),
                "token_estimate": obj.get("token_estimate", len(obj.get("prompt", "").split())),
            }
            teacher.append((hint, obj.get("teacher_answer", "")))
            if limit and len(teacher) >= limit:
                break
    return teacher


def build_semantic_hint(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scenario": task.get("scenario"),
        "output_shape": task.get("features", {}).get("output_shape"),
        "observation_text": task.get("prompt", ""),
        "has_numbers": any(ch.isdigit() for ch in task.get("prompt", "")),
        "token_estimate": len(task.get("prompt", "").split()),
    }


def build_teacher_tasks(core: ReasoningCore, tasks: List[Dict[str, Any]], limit: int) -> List[Tuple[Dict[str, Any], str]]:
    teacher: List[Tuple[Dict[str, Any], str]] = []
    for task in tasks:
        hint = build_semantic_hint(task)
        answer = core._try_solve(
            {
                "scenario": hint["scenario"],
                "output_shape": hint["output_shape"],
                "observation_text": hint["observation_text"],
            }
        )
        if answer:
            teacher.append((hint, answer))
        if len(teacher) >= limit:
            break
    return teacher


def train_epoch(core: ReasoningCore, teacher: List[Tuple[Dict[str, Any], str]], device: str, lr: float) -> float:
    if torch is None:
        raise RuntimeError("torch is required for training.")
    core.gru.train()  # type: ignore
    core.halt_head.train()  # type: ignore
    core.update_head.train()  # type: ignore
    params = list(core.gru.parameters()) + list(core.halt_head.parameters()) + list(core.update_head.parameters())
    if core.answer_head is not None:
        params += list(core.answer_head.parameters())
    optimizer = optim.Adam(params, lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    total_loss = 0.0

    for hint, answer_text in teacher:
        optimizer.zero_grad()
        features = core._featurize_vec(RelationalStateMock(), hint).to(device)  # type: ignore
        inp = features.unsqueeze(0).unsqueeze(0)
        hidden = torch.zeros(1, 1, core.config.hidden_size, device=device)
        out, hidden_out = core.gru(inp, hidden)
        halt_logit = core.halt_head(out).squeeze()
        update_logit = core.update_head(out).squeeze()
        target = torch.ones_like(halt_logit)
        halt_loss = bce(halt_logit, target)
        update_loss = bce(update_logit, target)
        answer_loss = 0.0
        if core.answer_head is not None:
            target_vec = core.encode_answer_text(answer_text, core.config.hidden_size).to(device)  # type: ignore
            pred_vec = core.answer_head(out).squeeze()
            answer_loss = mse(pred_vec, target_vec)
        loss = halt_loss + update_loss + answer_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(teacher))


class RelationalStateMock:
    """Minimal mock to satisfy featurizer input."""

    def __init__(self) -> None:
        self.entities = {}
        self.relations = []


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU reasoning core (minimal).")
    parser.add_argument("--tasks", type=str, default="dataset/test_prompts.json")
    parser.add_argument("--teacher", type=str, default="runs/data/teacher.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoints/core_gru.pt")
    args = parser.parse_args()

    if torch is None:
        print("Torch is not available; skipping training.")
        return

    device = "cpu"
    tasks = load_json_tasks(Path(args.tasks), args.limit)
    teacher = load_teacher_jsonl(Path(args.teacher), args.limit)
    reasoning_cfg = ReasoningConfig(use_rules=False, use_gru=True)
    core = ReasoningCore(config=reasoning_cfg)
    if hasattr(core, "to"):
        core.to(device)  # type: ignore
    if not teacher:
        teacher = build_teacher_tasks(core, tasks, args.limit)
    if not teacher:
        print("No teacher examples found; ensure scenarios match rule solvers.")
        return
    # Build answer bank
    if torch is not None:
        texts = []
        embs = []
        for _, ans in teacher:
            texts.append(ans)
            embs.append(core.encode_answer_text(ans, core.config.hidden_size))
        core.answer_bank_texts = texts
        core.answer_bank_embeddings = torch.stack(embs, dim=0)

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(core, teacher, device, args.lr)
        print(f"epoch {epoch+1}/{args.epochs} loss={loss:.4f}")

    core.save_checkpoint(args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory.relational_memory import RelationalUpdate


@dataclass
class Observation:
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ParsedTask:
    observation: Observation
    relational_update: "RelationalUpdate"
    structured: Optional[Dict[str, Any]] = None
    notes: Optional[Dict[str, Any]] = None


@dataclass
class StepTrace:
    step_idx: int
    core_state_summary: Dict[str, Any]
    relational_state_snapshot: Dict[str, Any]
    halt_probability: float
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "core_state_summary": self.core_state_summary,
            "relational_state_snapshot": self.relational_state_snapshot,
            "halt_probability": self.halt_probability,
            "notes": self.notes,
        }


@dataclass
class EpisodeTrace:
    steps: List[StepTrace]
    final_answer: str
    final_rel_state: Dict[str, Any]
    observation: Observation
    semantic_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": vars(self.observation),
            "final_answer": self.final_answer,
            "final_rel_state": self.final_rel_state,
            "steps": [step.to_dict() for step in self.steps],
            "semantic_info": self.semantic_info,
        }

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

from ..config.settings import RuntimeConfig
from ..config.reasoning import ReasoningConfig
from ..core.reasoning_core import CoreState, ReasoningCore
from ..memory.relational_memory import RelationalMemory
from ..semantic.engine import SemanticEngine
from ..types import EpisodeTrace, Observation, StepTrace


class Controller:
    def __init__(
        self,
        semantic_engine: Optional[SemanticEngine] = None,
        reasoning_core: Optional[ReasoningCore] = None,
        memory: Optional[RelationalMemory] = None,
        max_steps: int = 10,
        halt_threshold: float = 0.85,
        log_traces: bool = True,
        log_dir: Path = Path("runs"),
        reasoning_config: Optional[ReasoningConfig] = None,
    ) -> None:
        self.semantic_engine = semantic_engine or SemanticEngine()
        self.reasoning_config = reasoning_config or ReasoningConfig(max_steps=max_steps, halt_threshold=halt_threshold)
        self.reasoning_core = reasoning_core or ReasoningCore(config=self.reasoning_config)
        self._constraint_template = list(memory.state.constraints) if memory else []
        self.memory = RelationalMemory(constraints=self._constraint_template)
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        self.log_traces = log_traces
        self.log_dir = Path(log_dir)

    def run(self, observation: Observation) -> Tuple[str, EpisodeTrace]:
        # Reset per-episode memory to avoid entity collisions across runs.
        self.memory = RelationalMemory(constraints=self._constraint_template)
        parsed = self.semantic_engine.parse_events(observation)
        self.memory.apply_update(parsed.relational_update)

        steps: list[StepTrace] = []
        core_state: Optional[CoreState] = None
        semantic_info = {
            "semantic_source": parsed.structured.get("semantic_source"),
            "semantic_error": parsed.structured.get("error"),
            "model": self.semantic_engine.model_name,
        }

        for idx in range(self.max_steps):
            core_state, update, halt_probability = self.reasoning_core.step(
                self.memory.state,
                semantic_hint=parsed.structured,
                core_state=core_state,
            )
            self.memory.apply_update(update)
            base_notes = {
                "semantic_source": parsed.structured.get("semantic_source"),
                "semantic_error": parsed.structured.get("error"),
            }
            base_notes.update(update.metadata)
            step_trace = StepTrace(
                step_idx=idx,
                core_state_summary=core_state.summary(),
                relational_state_snapshot=self.memory.snapshot(),
                halt_probability=halt_probability,
                notes=base_notes,
            )
            steps.append(step_trace)
            if halt_probability >= self.halt_threshold:
                break

        final_answer = self.semantic_engine.render_final_answer(self.memory.state)
        episode_trace = EpisodeTrace(
            steps=steps,
            final_answer=final_answer,
            final_rel_state=self.memory.snapshot(),
            observation=observation,
            semantic_info=semantic_info,
        )
        if self.log_traces:
            self._log_trace(episode_trace)
        return final_answer, episode_trace

    @staticmethod
    def format_trace(trace: EpisodeTrace) -> str:
        lines = []
        lines.append(f"Observation: {trace.observation.text}")
        if trace.semantic_info:
            lines.append(f"Semantic: source={trace.semantic_info.get('semantic_source')}, error={trace.semantic_info.get('semantic_error')}")
        lines.append(f"Final answer: {trace.final_answer}")
        for step in trace.steps:
            lines.append(
                f"[{step.step_idx}] halt={step.halt_probability:.2f} "
                f"core={step.core_state_summary} notes={step.notes}"
            )
        return "\n".join(lines)

    def _log_trace(self, trace: EpisodeTrace) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.log_dir / f"trace_{timestamp}_{uuid4().hex}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, indent=2)


def build_default_controller(runtime_config: Optional[RuntimeConfig] = None) -> Controller:
    cfg = runtime_config or RuntimeConfig()
    return Controller(
        max_steps=cfg.max_steps,
        halt_threshold=cfg.halt_threshold,
        log_traces=cfg.log_traces,
        log_dir=cfg.log_dir,
    )

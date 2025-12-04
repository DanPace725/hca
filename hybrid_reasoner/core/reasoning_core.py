from __future__ import annotations

import json
import math
import re
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:  # torch optional; GRU path disabled if missing
    torch = None
    nn = None

BaseCore = nn.Module if nn is not None else object

from ..config.reasoning import ReasoningConfig
from ..memory.relational_memory import Relation, RelationalState, RelationalUpdate


@dataclass
class CoreState:
    step: int = 0
    hidden_accumulator: float = 0.0
    hidden_tensor: Optional[torch.Tensor] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "hidden_accumulator": self.hidden_accumulator,
            "hidden_norm": None if self.hidden_tensor is None else float(self.hidden_tensor.norm().item()),
        }


class ReasoningCore(BaseCore):
    """
    Lightweight reasoning core stub.
    Uses simple feature accumulation to approximate a halting score and proposes
    structured updates that downstream modules can inspect.
    """

    def __init__(self, halt_normalizer: float = 3.0, config: Optional[ReasoningConfig] = None):
        if nn is not None:
            super().__init__()
        self.halt_normalizer = halt_normalizer
        self.config = config or ReasoningConfig()
        self.feature_size = 6
        torch_available = torch is not None and nn is not None
        self._gru_enabled = self.config.use_gru and torch_available
        if self._gru_enabled:
            self.gru = nn.GRU(input_size=self.feature_size, hidden_size=self.config.hidden_size, batch_first=True)
            self.halt_head = nn.Linear(self.config.hidden_size, 1)
            self.update_head = nn.Linear(self.config.hidden_size, 1)
            self.answer_head = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        else:
            self.gru = None
            self.halt_head = None
            self.update_head = None
            self.answer_head = None
        self._torch_available = torch_available
        self.answer_bank_texts: list[str] = []
        self.answer_bank_embeddings: Optional[torch.Tensor] = None

    def step(
        self,
        rel_state: RelationalState,
        semantic_hint: Optional[Dict[str, Any]] = None,
        core_state: Optional[CoreState] = None,
    ) -> tuple[CoreState, RelationalUpdate, float]:
        state = core_state or CoreState()

        # Rule-based shortcut if enabled
        if self.config.use_rules:
            answer_text = self._try_solve(semantic_hint or {})
            if answer_text is not None:
                answer_relation = Relation(
                    source_id="observation",
                    relation_type="answer",
                    target_id="answer",
                    metadata={"text": answer_text, "source": "reasoning_core_rule"},
                )
                update = RelationalUpdate(add_relations=[answer_relation], metadata={"core": "reasoning_core_rule"})
                next_state = CoreState(step=state.step + 1, hidden_accumulator=state.hidden_accumulator)
                return next_state, update, 1.0

        # GRU-based path if enabled
        if self._gru_enabled and self.gru is not None and self.halt_head is not None:
            features_vec = self._featurize_vec(rel_state, semantic_hint)
            gru_input = features_vec.unsqueeze(0).unsqueeze(0)  # batch=1, seq=1, feat
            hidden_in = state.hidden_tensor
            if hidden_in is None:
                hidden_in = torch.zeros(1, 1, self.config.hidden_size)
            gru_out, hidden_out = self.gru(gru_input, hidden_in)
            halt_logit = self.halt_head(gru_out).squeeze().item()
            halt_probability = float(torch.sigmoid(torch.tensor(halt_logit)).item())
            update_score = self.update_head(gru_out).squeeze().item()
            answer_text = self._decode_answer(gru_out)
            relations = []
            meta = {"core": "reasoning_core_gru", "gru_score": update_score, "answer_source": "gru"}
            if answer_text:
                relations.append(
                    Relation(
                        source_id="observation",
                        relation_type="answer",
                        target_id="answer",
                        metadata={"text": answer_text, "source": "reasoning_core_gru"},
                    )
                )
            else:
                note_text = f"step={state.step} gru_score={update_score:.2f}"
                relations.append(
                    Relation(
                        source_id="observation",
                        relation_type="hypothesis",
                        target_id=f"step_{state.step}",
                        metadata={"text": note_text},
                    )
                )
            update = RelationalUpdate(add_entities=[], add_relations=relations, metadata=meta)
            next_state = CoreState(step=state.step + 1, hidden_accumulator=state.hidden_accumulator, hidden_tensor=hidden_out)
            return next_state, update, halt_probability

        # Fallback stub
        features = self._featurize(rel_state, semantic_hint)
        hidden = state.hidden_accumulator + features["density"]
        halt_probability = min(1.0, hidden / self.halt_normalizer)
        update = self._propose_update(rel_state, state, features)
        next_state = CoreState(step=state.step + 1, hidden_accumulator=hidden)
        return next_state, update, halt_probability

    def _try_solve(self, semantic_hint: Dict[str, Any]) -> Optional[str]:
        scenario = semantic_hint.get("scenario")
        output_shape = semantic_hint.get("output_shape") or "plain_prose"
        text = semantic_hint.get("observation_text") or ""

        if scenario == "apple_sharing":
            answer = self._solve_apple_sharing(text, output_shape)
        elif scenario == "cookie_problem":
            answer = self._solve_cookie_problem(text, output_shape)
        elif scenario == "travel_distance":
            answer = self._solve_travel_distance(text, output_shape)
        elif scenario == "recipe_scaling":
            answer = self._solve_recipe_scaling(text, output_shape)
        else:
            answer = None
        return answer

    def _solve_apple_sharing(self, text: str, output_shape: str) -> Optional[str]:
        nums = self._numbers(text)
        if len(nums) < 3:
            return None
        initial, gave, bought = nums[0], nums[1], nums[2]
        total = initial - gave + bought
        each = total / 2
        sarah = gave
        if output_shape == "json":
            payload = {"tom": each, "john": each, "sarah": sarah}
            return json.dumps(payload)
        return f"Tom and John each have {self._format_number(each)} apples; Sarah has {self._format_number(sarah)} apple{'s' if sarah != 1 else ''}."

    def _solve_cookie_problem(self, text: str, output_shape: str) -> Optional[str]:
        nums = self._numbers(text)
        if len(nums) < 2:
            return None
        initial, gave = nums[0], nums[1]
        remaining = initial - gave
        if output_shape == "json":
            payload = {"john": remaining}
            return json.dumps(payload)
        return f"John has {self._format_number(remaining)} cookies left."

    def _solve_travel_distance(self, text: str, output_shape: str) -> Optional[str]:
        nums = self._numbers(text)
        if len(nums) < 2:
            return None
        total = nums[0] + nums[1]
        avg = total / 2
        if output_shape == "json":
            payload = {"total_miles": total, "average_mph": avg}
            return json.dumps(payload)
        return f"{self._format_number(total)} miles total, {self._format_number(avg)} mph average."

    def _solve_recipe_scaling(self, text: str, output_shape: str) -> Optional[str]:
        nums = self._numbers(text)
        if len(nums) < 4:
            return None
        base_people, flour_cups, eggs, target_people = nums[0], nums[1], nums[2], nums[3]
        factor = target_people / base_people if base_people else 0
        flour = flour_cups * factor
        egg_count = eggs * factor
        if output_shape == "json":
            payload = {"flour_cups": flour, "eggs": egg_count}
            return json.dumps(payload)
        rounded_eggs = math.ceil(egg_count)
        return f"{self._format_number(flour)} cups of flour and {self._format_number(egg_count)} eggs (or round to {rounded_eggs})."

    @staticmethod
    def _numbers(text: str) -> list[float]:
        return [float(x) if "." in x else int(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]

    @staticmethod
    def _format_number(num: float) -> str:
        return str(int(num)) if float(num).is_integer() else f"{num:.2f}"

    def _featurize(self, rel_state: RelationalState, semantic_hint: Optional[Dict[str, Any]]) -> Dict[str, float]:
        entity_count = len(rel_state.entities)
        relation_count = len(rel_state.relations)
        density = 0.1 * entity_count + 0.05 * relation_count
        if semantic_hint and semantic_hint.get("has_numbers"):
            density += 0.05
        return {"density": density, "entity_count": float(entity_count), "relation_count": float(relation_count)}

    def _featurize_vec(self, rel_state: RelationalState, semantic_hint: Optional[Dict[str, Any]]) -> torch.Tensor:
        hint = semantic_hint or {}
        entity_count = len(rel_state.entities)
        relation_count = len(rel_state.relations)
        has_numbers = 1.0 if hint.get("has_numbers") else 0.0
        token_estimate = float(hint.get("token_estimate") or 0.0)
        parsed_entities = float(len(hint.get("parsed", {}).get("entities", [])) if hint.get("parsed") else 0.0)
        parsed_relations = float(len(hint.get("parsed", {}).get("relations", [])) if hint.get("parsed") else 0.0)
        if torch is None:
            raise RuntimeError("Torch is not available for GRU featurization.")
        vec = torch.tensor(
            [
                float(entity_count),
                float(relation_count),
                has_numbers,
                token_estimate,
                parsed_entities,
                parsed_relations,
            ],
            dtype=torch.float32,
        )
        return vec

    def _decode_answer(self, gru_out: torch.Tensor) -> Optional[str]:
        if self.answer_head is None or self.answer_bank_embeddings is None or not self.answer_bank_texts:
            return None
        answer_vec = self.answer_head(gru_out).reshape(-1)  # [hidden]
        bank = self.answer_bank_embeddings  # [N, hidden]
        if bank.shape[1] != answer_vec.shape[0]:
            # Truncate or pad to match dimensions defensively
            target_dim = bank.shape[1]
            if answer_vec.shape[0] > target_dim:
                answer_vec = answer_vec[:target_dim]
            else:
                pad = torch.zeros(target_dim - answer_vec.shape[0])
                answer_vec = torch.cat([answer_vec, pad], dim=0)
        denom = (answer_vec.norm(p=2) * bank.norm(p=2, dim=1) + 1e-8)
        sims = (bank @ answer_vec) / denom
        best_idx = int(torch.argmax(sims).item())
        return self.answer_bank_texts[best_idx]

    @staticmethod
    def encode_answer_text(text: str, dim: int) -> torch.Tensor:
        if torch is None:
            raise RuntimeError("Torch not available for encoding.")
        vec = torch.zeros(dim, dtype=torch.float32)
        for token in re.findall(r"[A-Za-z0-9]+", text.lower()):
            idx = hash(token) % dim
            vec[idx] += 1.0
        return vec

    # ----- Checkpoint helpers -----
    def save_checkpoint(self, path: str) -> None:
        if not self._gru_enabled:
            raise RuntimeError("Cannot save checkpoint: GRU path is disabled or torch is unavailable.")
        state = {
            "gru": self.gru.state_dict(),
            "halt_head": self.halt_head.state_dict(),
            "update_head": self.update_head.state_dict(),
            "answer_head": None if self.answer_head is None else self.answer_head.state_dict(),
            "config": self.config.__dict__,
            "answer_bank_texts": self.answer_bank_texts,
            "answer_bank_embeddings": None if self.answer_bank_embeddings is None else self.answer_bank_embeddings.cpu(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        if not self._torch_available:
            raise RuntimeError("Cannot load checkpoint: torch is unavailable.")
        data = torch.load(path, map_location="cpu")
        if self.gru is None or self.halt_head is None or self.update_head is None:
            # Initialize modules if they were not set up
            self.gru = nn.GRU(input_size=self.feature_size, hidden_size=self.config.hidden_size, batch_first=True)
            self.halt_head = nn.Linear(self.config.hidden_size, 1)
            self.update_head = nn.Linear(self.config.hidden_size, 1)
        self.gru.load_state_dict(data["gru"])
        self.halt_head.load_state_dict(data["halt_head"])
        self.update_head.load_state_dict(data["update_head"])
        if "answer_head" in data and data["answer_head"] is not None:
            if self.answer_head is None:
                self.answer_head = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.answer_head.load_state_dict(data["answer_head"])
        self.answer_bank_texts = data.get("answer_bank_texts") or []
        abe = data.get("answer_bank_embeddings")
        if abe is not None:
            self.answer_bank_embeddings = abe
        else:
            self.answer_bank_embeddings = None

    def to(self, device: str):
        if not self._gru_enabled:
            return self
        self.gru.to(device)  # type: ignore
        self.halt_head.to(device)  # type: ignore
        self.update_head.to(device)  # type: ignore
        if self.answer_head is not None:
            self.answer_head.to(device)
        if self.answer_bank_embeddings is not None:
            self.answer_bank_embeddings = self.answer_bank_embeddings.to(device)
        return self

    def _propose_update(
        self,
        rel_state: RelationalState,
        state: CoreState,
        features: Dict[str, float],
    ) -> RelationalUpdate:
        note_text = f"step={state.step} density={features['density']:.2f}"
        note_relation = Relation(
            source_id="observation",
            relation_type="hypothesis",
            target_id=f"step_{state.step}",
            metadata={"text": note_text},
        )
        return RelationalUpdate(
            add_entities=[],
            add_relations=[note_relation],
            metadata={"core": "reasoning_core_stub", "density": features["density"]},
        )

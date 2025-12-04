from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple
from urllib import error, request

from ..config.settings import ModelConfig
from ..memory.relational_memory import Entity, Relation, RelationalState, RelationalUpdate
from ..types import Observation, ParsedTask


class SemanticEngine:
    """
    Ollama-backed semantic utilities with a safe fallback stub when the backend is unavailable.
    """

    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        self.model_config = model_config or ModelConfig()
        self.model_name = self.model_config.model_name

    def parse_events(self, observation: Observation) -> ParsedTask:
        normalized_text = observation.text.strip()
        response_text, err = self._call_model(
            prompt=self._build_event_prompt(normalized_text),
            system="You are a precise semantic parser. Return valid JSON only.",
        )
        parsed_update, structured, source = self._parse_response_to_update(response_text, normalized_text, err)
        if observation.metadata:
            structured["scenario"] = observation.metadata.get("scenario")
            structured["output_shape"] = observation.metadata.get("output_shape")
            structured["observation_text"] = normalized_text

        return ParsedTask(
            observation=observation,
            relational_update=parsed_update,
            structured=structured,
            notes={"model": self.model_name, "source": source},
        )

    def summarize_state(self, rel_state: RelationalState) -> str:
        entity_count = len(rel_state.entities)
        relation_count = len(rel_state.relations)
        return f"Memory holds {entity_count} entities and {relation_count} relations."

    def render_final_answer(self, rel_state: RelationalState) -> str:
        candidate = self._extract_answer(rel_state)
        if candidate:
            return candidate

        summary = self.summarize_state(rel_state)
        prompt = (
            "You are a precise summarizer. Given a relational memory snapshot, render a concise answer. "
            "Use only the provided JSON; do not invent facts.\n"
            f"REL_STATE_JSON:\n{json.dumps(rel_state.to_dict())}\n\nAnswer:"
        )
        response, _ = self._call_model(prompt=prompt, system="Render final answer.")
        if response:
            return response.strip()
        return summary

    def _extract_answer(self, rel_state: RelationalState) -> Optional[str]:
        for relation in reversed(rel_state.relations):
            if relation.relation_type in {"answer", "hypothesis"}:
                text = relation.metadata.get("text") or relation.metadata.get("value")
                if text:
                    return str(text)
        return None

    def _call_model(self, prompt: str, system: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        if len(prompt) > self.model_config.max_prompt_chars:
            return None, f"prompt_too_long_{len(prompt)}"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.model_config.temperature,
                "num_predict": self.model_config.max_tokens,
                "stop": list(self.model_config.stop_tokens),
            },
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.model_config.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.model_config.request_timeout) as resp:
                content = resp.read().decode("utf-8")
                parsed = json.loads(content)
                return parsed.get("response"), None
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError, ConnectionError) as exc:
            return None, str(exc)

    def _parse_response_to_update(
        self, response_text: Optional[str], normalized_text: str, error_note: Optional[str]
    ) -> Tuple[RelationalUpdate, Dict[str, Any], str]:
        raw_response = response_text
        if response_text:
            json_text = self._repair_json(self._extract_json(response_text))
            if json_text:
                try:
                    payload = json.loads(json_text)
                    validated, errors = self._validate_payload(payload)
                    update = self._update_from_payload(payload, normalized_text)
                    structured = {
                        "token_estimate": len(normalized_text.split()),
                        "has_numbers": any(char.isdigit() for char in normalized_text),
                        "parsed": validated,
                        "parse_errors": errors,
                        "semantic_source": "semantic_ollama",
                    }
                    if self.model_config.log_raw_response:
                        structured["raw_response"] = raw_response
                    return update, structured, "semantic_ollama"
                except json.JSONDecodeError as e:
                    error_note = f"Ollama returned invalid JSON: {str(e)}"

        # Fallback or fail-fast
        stub_structured = {
            "token_estimate": len(normalized_text.split()),
            "has_numbers": any(char.isdigit() for char in normalized_text),
            "semantic_source": "semantic_stub",
            "error": error_note,
        }
        if self.model_config.log_raw_response and raw_response:
            stub_structured["raw_response"] = raw_response
        if not self.model_config.fallback_to_stub:
            raise RuntimeError(
                f"Semantic parsing FAILED: {error_note or 'bad_json'}; raw_response={raw_response}"
            )

        entity = Entity(
            id="observation",
            type="text",
            metadata={"text": normalized_text, "length": len(normalized_text)},
        )
        relation = Relation(
            source_id=entity.id,
            relation_type="describes",
            target_id="observation_text",
            metadata={"token_estimate": len(normalized_text.split())},
        )
        update = RelationalUpdate(
            add_entities=[entity],
            add_relations=[relation],
            metadata={"source": "semantic_stub", "error": error_note or "ollama_response_missing"},
        )
        return update, stub_structured, "semantic_stub"

    def _update_from_payload(self, payload: Dict[str, Any], normalized_text: str) -> RelationalUpdate:
        entities_data = payload.get("entities") or []
        relations_data = payload.get("relations") or []

        entities = []
        for idx, ent in enumerate(entities_data):
            ent_id = str(ent.get("id") or f"ent_{idx}")
            ent_type = str(ent.get("type") or "unknown")
            metadata = ent.get("metadata") or {}
            entities.append(Entity(id=ent_id, type=ent_type, metadata=metadata))

        relations = []
        for rel in relations_data:
            source = str(rel.get("source_id") or "observation")
            target = str(rel.get("target_id") or "observation_text")
            rel_type = str(rel.get("relation_type") or "relates_to")
            metadata = rel.get("metadata") or {}
            relations.append(Relation(source_id=source, relation_type=rel_type, target_id=target, metadata=metadata))

        if not entities:
            entities.append(
                Entity(
                    id="observation",
                    type="text",
                    metadata={"text": normalized_text, "length": len(normalized_text)},
                )
            )
        return RelationalUpdate(
            add_entities=entities,
            add_relations=relations,
            metadata={"source": "semantic_ollama"},
        )

    def _validate_payload(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
        errors: list[str] = []
        entities = payload.get("entities")
        relations = payload.get("relations")
        if entities is None or not isinstance(entities, list):
            errors.append("missing_or_invalid_entities")
            entities = []
        if relations is None or not isinstance(relations, list):
            errors.append("missing_or_invalid_relations")
            relations = []
        cleaned = {"entities": entities, "relations": relations}
        return cleaned, errors

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        # First, try to extract JSON from markdown code blocks
        code_block_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1)
        # Fallback: Find any JSON object; take the first balanced one
        brace_count = 0
        start_idx = None
        for i, ch in enumerate(text):
            if ch == "{":
                brace_count += 1
                if start_idx is None:
                    start_idx = i
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    return text[start_idx : i + 1]
        return None

    @staticmethod
    def _repair_json(json_str: Optional[str]) -> Optional[str]:
        if not json_str:
            return None
        json_str = re.sub(r"```(?:json)?", "", json_str).strip()
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
        json_str = re.sub(r"\}\s*\{", "}, {", json_str)
        brace_count = 0
        last_valid_pos = -1
        for i, char in enumerate(json_str):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    last_valid_pos = i + 1
        if last_valid_pos > 0:
            json_str = json_str[:last_valid_pos]
        return json_str

    @staticmethod
    def _build_event_prompt(normalized_text: str) -> str:
        return (
            "Extract entities and relations from the text. "
            "Respond ONLY with valid JSON. No explanations or markdown.\n\n"
            "Schema:\n"
            '{\n'
            '  "entities": [\n'
            '    {"id": "tom", "type": "person", "metadata": {}},\n'
            '    {"id": "sarah", "type": "person", "metadata": {}}\n'
            '  ],\n'
            '  "relations": [\n'
            '    {"source_id": "tom", "relation_type": "gives", "target_id": "sarah", "metadata": {"item": "apple", "quantity": 1}}\n'
            '  ]\n'
            '}\n\n'
            "Rules:\n"
            "- Use short lowercase ids (names, objects)\n"
            "- All JSON must be properly closed with matching braces\n"
            "- Only extract facts explicitly stated in the text\n"
            "- Ensure all objects have matching opening and closing braces\n\n"
            "TEXT:\n"
            f"{normalized_text}\n\n"
            "JSON:"
        )

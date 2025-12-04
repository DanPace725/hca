from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Entity:
    id: str
    type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    source_id: str
    relation_type: str
    target_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Constraint:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationalUpdate:
    add_entities: List[Entity] = field(default_factory=list)
    add_relations: List[Relation] = field(default_factory=list)
    remove_relations: List[Relation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationalState:
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": {k: vars(v) for k, v in self.entities.items()},
            "relations": [vars(r) for r in self.relations],
            "constraints": [vars(c) for c in self.constraints],
        }

    def clone(self) -> "RelationalState":
        return RelationalState(
            entities={k: Entity(**vars(v)) for k, v in self.entities.items()},
            relations=[Relation(**vars(r)) for r in self.relations],
            constraints=[Constraint(**vars(c)) for c in self.constraints],
        )


class RelationalMemory:
    def __init__(self, constraints: Optional[List[Constraint]] = None):
        constraint_list = constraints or [Constraint(name="quantity_nonnegative")]
        self.state = RelationalState(constraints=list(constraint_list))

    def add_entity(self, entity: Entity) -> None:
        if entity.id in self.state.entities:
            raise ValueError(f"Duplicate entity id: {entity.id}")
        self._apply_quantity_checks(entity.metadata)
        self.state.entities[entity.id] = entity

    def add_relation(self, relation: Relation) -> None:
        if relation.source_id not in self.state.entities or relation.target_id not in self.state.entities:
            relation.metadata.setdefault("warnings", []).append("dangling_relation")
        self._apply_quantity_checks(relation.metadata)
        self.state.relations.append(relation)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.state.entities.get(entity_id)

    def get_relations_for(self, entity_id: str) -> List[Relation]:
        return [r for r in self.state.relations if r.source_id == entity_id or r.target_id == entity_id]

    def apply_update(self, update: RelationalUpdate) -> None:
        for entity in update.add_entities:
            self.add_entity(entity)
        for relation in update.add_relations:
            self.add_relation(relation)
        if update.remove_relations:
            self._remove_relations(update.remove_relations)

    def snapshot(self) -> Dict[str, Any]:
        return self.state.to_dict()

    def _remove_relations(self, relations: List[Relation]) -> None:
        to_remove = {self._relation_key(r) for r in relations}
        self.state.relations = [r for r in self.state.relations if self._relation_key(r) not in to_remove]

    @staticmethod
    def _relation_key(relation: Relation) -> tuple:
        return (relation.source_id, relation.relation_type, relation.target_id)

    def _apply_quantity_checks(self, metadata: Dict[str, Any]) -> None:
        if not self._has_constraint("quantity_nonnegative"):
            return
        qty = metadata.get("quantity")
        if qty is None:
            return
        try:
            val = float(qty)
        except (TypeError, ValueError):
            metadata.setdefault("warnings", []).append("quantity_unparsed")
            return
        if val < 0:
            metadata.setdefault("warnings", []).append("negative_quantity")

    def _has_constraint(self, name: str) -> bool:
        return any(c.name == name for c in self.state.constraints)

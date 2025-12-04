from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReasoningConfig:
    use_rules: bool = False  # Disabled by default - rules bypass learning
    use_gru: bool = True
    hidden_size: int = 64
    halt_threshold: float = 0.85
    max_steps: int = 10


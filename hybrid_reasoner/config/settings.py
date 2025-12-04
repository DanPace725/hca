from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "phi3:latest"  # Ollama model tag
    model_path: Optional[Path] = None
    endpoint: str = "http://localhost:11434/api/generate"
    temperature: float = 0.1  # Slight randomness helps with JSON formatting
    max_tokens: int = 512  # More room for complete JSON responses
    request_timeout: int = 90
    max_prompt_chars: int = 8000  # Guardrail for oversized prompts
    fallback_to_stub: bool = False  # Fail fast by default
    log_raw_response: bool = True
    stop_tokens: tuple[str, ...] = ("\n\n", "###", "```")


@dataclass
class RuntimeConfig:
    max_steps: int = 10
    halt_threshold: float = 0.85
    log_traces: bool = True
    log_dir: Path = Path("runs")

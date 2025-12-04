# Decisions for Hybrid Reasoner Setup (all open items resolved)

- Backend: Use **Ollama** as the runner for Phi-3 (available in `G:\Ollama\`, model present as `phi3:latest`). Direct transformers path exists at `G:\Phi3\`; llama.cpp is not required.
- Evaluation set: Start with `dataset/test_prompts.json`; reshape as needed.
- Harness fields: Minimum required per task — `test_id`, `prompt`, `expected_answer`, and `features.output_shape` (e.g., `json` vs `plain_prose`). Optional but useful: `scenario`.
- Logging: Persist per-episode traces as JSON under `runs/` by default.
- Constraints: Keep MVP checks minimal—duplicate IDs and dangling relations only.
- Halting defaults: Use conservative settings for MVP (current defaults: `max_steps=10`, `halt_threshold=0.85`).

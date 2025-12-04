import argparse
import json
from pathlib import Path

from hybrid_reasoner import Observation, build_default_controller
from hybrid_reasoner.config.settings import RuntimeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hybrid reasoning scaffold.")
    parser.add_argument("text", help="Input text to process.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=RuntimeConfig.max_steps,
        help=f"Maximum reasoning steps (default {RuntimeConfig.max_steps}).",
    )
    parser.add_argument(
        "--halt-threshold",
        type=float,
        default=RuntimeConfig.halt_threshold,
        help=f"Halting threshold (default {RuntimeConfig.halt_threshold}).",
    )
    parser.add_argument(
        "--no-log-trace",
        action="store_true",
        help="Disable writing traces to disk.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(RuntimeConfig.log_dir),
        help="Directory for JSON trace logs.",
    )
    parser.add_argument("--show-trace", action="store_true", help="Print structured trace output.")
    parser.add_argument(
        "--pretty-trace",
        action="store_true",
        help="Print a human-readable trace summary.",
    )
    parser.add_argument(
        "--use-rules",
        action="store_true",
        help="Enable rule-based shortcuts (disabled by default - for comparison testing only).",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Use ONLY rules, disable GRU (for comparison testing).",
    )
    parser.add_argument(
        "--core-checkpoint",
        type=str,
        default=None,
        help="Path to a GRU core checkpoint to load (optional).",
    )
    parser.add_argument(
        "--semantic-fallback",
        action="store_true",
        help="Allow semantic engine to fallback to stub on parse failure.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario type for rule-based solving (e.g., apple_sharing, cookie_problem, travel_distance, recipe_scaling).",
    )
    args = parser.parse_args()

    runtime_config = RuntimeConfig(
        max_steps=args.max_steps,
        halt_threshold=args.halt_threshold,
        log_traces=not args.no_log_trace,
        log_dir=Path(args.log_dir),
    )
    from hybrid_reasoner.config.reasoning import ReasoningConfig
    reasoning_cfg = ReasoningConfig(
        use_rules=args.use_rules or args.rules_only,  # Disabled by default
        use_gru=not args.rules_only,  # Enabled by default unless rules-only
        max_steps=runtime_config.max_steps,
        halt_threshold=runtime_config.halt_threshold,
    )
    controller = build_default_controller(runtime_config)
    controller.reasoning_config = reasoning_cfg
    controller.reasoning_core.config = reasoning_cfg
    if args.core_checkpoint:
        try:
            controller.reasoning_core.load_checkpoint(args.core_checkpoint)
            controller.reasoning_config.use_gru = True
            controller.reasoning_core.config.use_gru = True
        except Exception as exc:
            print(f"Warning: failed to load checkpoint: {exc}")

    # Build observation with optional scenario metadata
    obs_metadata = {}
    if args.scenario:
        obs_metadata["scenario"] = args.scenario
    
    answer, trace = controller.run(Observation(text=args.text, metadata=obs_metadata))
    print(f"Final answer: {answer}")

    if args.show_trace:
        print(json.dumps(trace.to_dict(), indent=2))
    if args.pretty_trace:
        print("--- TRACE ---")
        print(controller.format_trace(trace))


if __name__ == "__main__":
    main()

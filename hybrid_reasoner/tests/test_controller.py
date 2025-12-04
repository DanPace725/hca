import unittest

from hybrid_reasoner import Observation, build_default_controller
from hybrid_reasoner.config.reasoning import ReasoningConfig
from hybrid_reasoner.config.settings import ModelConfig
from hybrid_reasoner.semantic.engine import SemanticEngine


class ControllerIntegrationTest(unittest.TestCase):
    def test_run_produces_trace(self) -> None:
        from hybrid_reasoner.config.settings import RuntimeConfig

        semantic = SemanticEngine(ModelConfig(fallback_to_stub=True))
        controller = build_default_controller(runtime_config=RuntimeConfig(log_traces=False))
        controller.semantic_engine = semantic
        answer, trace = controller.run(Observation(text="Alice has 3 apples."))

        self.assertIsInstance(answer, str)
        self.assertGreaterEqual(len(trace.steps), 1)
        self.assertIn("entities", trace.final_rel_state)
        self.assertEqual(trace.observation.text, "Alice has 3 apples.")

    def test_can_disable_logging(self) -> None:
        from hybrid_reasoner.config.settings import RuntimeConfig

        runtime = RuntimeConfig(log_traces=False)
        controller = build_default_controller(runtime)
        controller.semantic_engine = SemanticEngine(ModelConfig(fallback_to_stub=True))
        _, trace = controller.run(Observation(text="Bob buys 2 oranges."))
        self.assertEqual(trace.observation.text, "Bob buys 2 oranges.")

    def test_rule_only_mode(self) -> None:
        runtime = None
        controller = build_default_controller(runtime)
        controller.reasoning_config = ReasoningConfig(use_rules=True, use_gru=False)
        controller.reasoning_core.config = controller.reasoning_config
        controller.semantic_engine = SemanticEngine(ModelConfig(fallback_to_stub=True))
        answer, trace = controller.run(
            Observation(text="John has 5 cookies and gives 2 to Mary.", metadata={"scenario": "cookie_problem"})
        )
        self.assertIn("cookies", answer)
        self.assertGreaterEqual(len(trace.steps), 1)


if __name__ == "__main__":
    unittest.main()

import unittest

from evg.config import EngineConfig
from evg.engine import EVGEngine
from evg.models import get_model_registry
from evg.types import TaskType


class ModelRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = get_model_registry()

    def test_requested_model_families_are_registered(self) -> None:
        for model_id in (
            "wan2.2",
            "hunyuanvideo-1.5",
            "lingbot-video",
            "longcat-video",
            "cosmos3",
            "skyreels-v3",
            "bernini",
        ):
            with self.subTest(model_id=model_id):
                self.assertEqual(self.registry.get(model_id).id, model_id)

    def test_alias_lookup(self) -> None:
        self.assertEqual(self.registry.get("wan22").id, "wan2.2")
        self.assertEqual(self.registry.get("hunyuan").id, "hunyuanvideo-1.5")
        self.assertEqual(self.registry.get("cosmos").id, "cosmos3")

    def test_text_to_video_filter_contains_expected_baselines(self) -> None:
        ids = {spec.id for spec in self.registry.list(task=TaskType.TEXT_TO_VIDEO)}
        self.assertIn("wan2.2", ids)
        self.assertIn("hunyuanvideo-1.5", ids)
        self.assertIn("longcat-video", ids)

    def test_engine_can_create_runtime_plan(self) -> None:
        engine = EVGEngine(EngineConfig(model="wan2.2", variant="t2v-a14b"))
        plan = engine.plan()
        self.assertEqual(plan.model_id, "wan2.2")
        self.assertEqual(plan.variant_id, "t2v-a14b")
        self.assertTrue(plan.can_execute)


if __name__ == "__main__":
    unittest.main()

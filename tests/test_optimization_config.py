import json
import tempfile
import unittest
from pathlib import Path

from evg.config import SparsityRule, SparsityScheduleConfig


class SparsityScheduleConfigTest(unittest.TestCase):
    def test_dense_prefix_then_uniform_sparse(self) -> None:
        schedule = SparsityScheduleConfig(
            enabled=True,
            dense_step_fraction=0.25,
            default_sparsity=0.8,
        ).resolve(total_steps=8, num_layers=3)

        self.assertEqual(schedule.dense_steps, 2)
        self.assertEqual(schedule.ratios[:2], ((0.0, 0.0, 0.0),) * 2)
        self.assertEqual(schedule.ratios[2:], ((0.8, 0.8, 0.8),) * 6)

    def test_ordered_rules_support_step_and_layer_overrides(self) -> None:
        config = SparsityScheduleConfig(
            enabled=True,
            dense_step_fraction=0.25,
            default_sparsity=0.8,
            rules=(
                SparsityRule(layers="0", sparsity=0.4),
                SparsityRule(steps="4-5", layers="1,2", sparsity=0.6),
                SparsityRule(steps=5, layers=2, sparsity=0.7),
            ),
        )

        schedule = config.resolve(total_steps=8, num_layers=3)
        self.assertEqual(schedule.sparsities_for_step(0), (0.0, 0.0, 0.0))
        self.assertEqual(schedule.sparsities_for_step(2), (0.4, 0.8, 0.8))
        self.assertEqual(schedule.sparsities_for_step(4), (0.4, 0.6, 0.6))
        self.assertEqual(schedule.sparsities_for_step(5), (0.4, 0.6, 0.7))

    def test_dense_prefix_cannot_be_overridden(self) -> None:
        schedule = SparsityScheduleConfig(
            enabled=True,
            dense_step_fraction=0.5,
            default_sparsity=0.8,
            rules=(SparsityRule(steps=0, layers="*", sparsity=0.9),),
        ).resolve(total_steps=4, num_layers=2)

        self.assertTrue(schedule.is_dense_step(0))
        self.assertTrue(schedule.is_dense_step(1))
        self.assertEqual(schedule.sparsities_for_step(2), (0.8, 0.8))

    def test_full_matrix_supports_every_step_layer_cell(self) -> None:
        schedule = SparsityScheduleConfig(
            enabled=True,
            dense_step_fraction=0.0,
            sparsity_matrix=((0.1, 0.2), (0.3, 0.4), (0.5, 0.6)),
        ).resolve(total_steps=3, num_layers=2)

        self.assertEqual(schedule.sparsity_for(0, 1), 0.2)
        self.assertEqual(schedule.sparsity_for(2, 0), 0.5)

    def test_json_loader_accepts_compact_selectors(self) -> None:
        payload = {
            "dense_step_fraction": 0.25,
            "default_sparsity": 0.8,
            "rules": [
                {"steps": "2-3", "layers": "0,2-3", "sparsity": 0.65},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = SparsityScheduleConfig.from_json_file(path, enabled=True)

        schedule = config.resolve(total_steps=4, num_layers=4)
        self.assertEqual(schedule.sparsities_for_step(2), (0.65, 0.8, 0.65, 0.65))

    def test_disabled_schedule_resolves_to_dense(self) -> None:
        schedule = SparsityScheduleConfig(
            enabled=False,
            dense_step_fraction=0.0,
            default_sparsity=0.8,
        ).resolve(total_steps=2, num_layers=2)
        self.assertEqual(schedule.ratios, ((0.0, 0.0), (0.0, 0.0)))

    def test_runtime_dimensions_validate_matrix_and_rules(self) -> None:
        matrix = SparsityScheduleConfig(
            enabled=True,
            sparsity_matrix=((0.5,),),
        )
        with self.assertRaisesRegex(ValueError, "row count"):
            matrix.resolve(total_steps=2, num_layers=1)

        rule = SparsityScheduleConfig(
            enabled=True,
            rules=(SparsityRule(layers=4, sparsity=0.5),),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            rule.resolve(total_steps=2, num_layers=4)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from anemoi.models.minimax_h3.runner import (
    _configure_cuda_driver_link,
    _enable_long_sequence_group_offload,
)
from anemoi.models.minimax_h3.runtime import (
    EvalTimer,
    OffloadTransferTimer,
    _load_local_components,
)


class _Pipeline:
    def __init__(self, *, register: bool = True) -> None:
        self.register = register
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def load_components(self, names: list[str], **kwargs: object) -> None:
        self.calls.append((names, kwargs))
        if self.register:
            for name in names:
                setattr(self, name, object())


class LocalComponentLoadingTests(unittest.TestCase):
    def test_triton_driver_directory_exposes_linker_name_and_soname(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver_root = root / "driver"
            driver_root.mkdir()
            driver = driver_root / "libcuda.so.1"
            driver.touch()
            with patch.dict(os.environ, {"LD_LIBRARY_PATH": str(driver_root)}):
                _configure_cuda_driver_link(root / "cache")

            link_root = root / "cache/driver-lib"
            self.assertEqual((link_root / "libcuda.so").resolve(), driver)
            self.assertEqual((link_root / "libcuda.so.1").resolve(), driver)

    def test_components_are_loaded_from_the_prepared_tree_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vae").mkdir()
            (root / "audio_vae").mkdir()
            pipeline = _Pipeline()
            with patch.dict(os.environ, {"H3_MODEL_ROOT": str(root)}):
                _load_local_components(pipeline, ["vae", "audio_vae"])

            self.assertEqual(len(pipeline.calls), 1)
            names, kwargs = pipeline.calls[0]
            self.assertEqual(names, ["vae", "audio_vae"])
            self.assertEqual(kwargs["pretrained_model_name_or_path"], str(root.resolve()))
            self.assertIs(kwargs["local_files_only"], True)

    def test_missing_component_directory_fails_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _Pipeline()
            with patch.dict(os.environ, {"H3_MODEL_ROOT": temporary}):
                with self.assertRaisesRegex(FileNotFoundError, "audio_vae"):
                    _load_local_components(pipeline, ["audio_vae"])
            self.assertEqual(pipeline.calls, [])


class LongSequenceTimingTests(unittest.TestCase):
    def test_group_offload_is_long_sequence_only(self) -> None:
        class Transformer:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def enable_group_offload(self, **kwargs: object) -> None:
                self.calls.append(kwargs)

        transformer = Transformer()
        self.assertIsNone(_enable_long_sequence_group_offload(transformer, 38_247))
        self.assertEqual(transformer.calls, [])

        result = _enable_long_sequence_group_offload(transformer, 73_768)
        self.assertEqual(result["num_blocks_per_group"], 1)
        self.assertEqual(result["offload_device"], "cpu")
        self.assertEqual(transformer.calls, [result])

    def test_transformer_timer_excludes_group_transfer_spans(self) -> None:
        class Group:
            def onload_(self) -> None:
                pass

            def offload_(self) -> None:
                pass

        group = Group()

        class Transformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = torch.nn.Identity()
                hook = SimpleNamespace(group=group)
                self.block._diffusers_hook = SimpleNamespace(hooks={"group": hook})

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                group.onload_()
                result = value + 1
                group.offload_()
                return result

        durations = (10.0, 2.0, 3.0)

        class Event:
            created = 0

            def __init__(self, **_: object) -> None:
                self.index = Event.created
                Event.created += 1

            def record(self) -> None:
                pass

            def elapsed_time(self, _: object) -> float:
                return durations[self.index // 2]

        transformer = Transformer()
        transfers = OffloadTransferTimer(transformer)
        timer = EvalTimer(transformer, excluded=transfers)
        with patch("torch.cuda.Event", Event), patch("torch.cuda.synchronize"):
            transfers.install()
            timer.install()
            self.assertEqual(timer.module(torch.tensor(1)), torch.tensor(2))
            timer.remove()
            transfers.remove()

            self.assertEqual(timer.resident_samples_ms(), [10.0])
            self.assertEqual(timer.excluded_samples_ms(), [5.0])
            self.assertEqual(timer.samples_ms(), [5.0])

    def test_silently_unloaded_component_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vae").mkdir()
            pipeline = _Pipeline(register=False)
            with patch.dict(os.environ, {"H3_MODEL_ROOT": temporary}):
                with self.assertRaisesRegex(RuntimeError, "vae"):
                    _load_local_components(pipeline, ["vae"])


if __name__ == "__main__":
    unittest.main()

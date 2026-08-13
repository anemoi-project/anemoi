from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evg.models.minimax_h3.runtime import _load_local_components


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

"""Unit tests for the modular notebook generation subsystem (Phase 9 / Gate 9).

Validates cell builders, registry metadata resolution, backward compatibility,
and exact structural parity against baseline notebooks for mirnet_exposure and forex_predictor.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

import training.notebook_generator as legacy
from training.ecosystem import TRAINING_SERVICE, ecosystem
from training.notebooks import (
    ModelNotebookMeta,
    generate_colab_inference_notebook,
    generate_colab_training_notebook,
    generate_colab_usage_notebook,
    generate_inference_notebook,
    generate_training_notebook,
    generate_usage_notebook,
    load_registry,
    resolve_model_metadata,
)
from training.notebooks.cells.base import make_code_cell, make_markdown_cell
from training.notebooks.cells.data import build_symlink_cell
from training.notebooks.cells.deps import build_install_cell
from training.notebooks.cells.env import (
    _build_env_var_lines,
    _load_runtime_env,
    build_sentinel_cell,
)
from training.notebooks.cells.inference import (
    build_onnx_fp16_cell,
    build_onnx_fp32_cell,
    build_pth_cell,
)
from training.notebooks.cells.model import (
    build_checkpoint_recovery_cell,
    build_hub_prep_cell,
    build_stealth_cell,
)
from training.notebooks.cells.repo import build_clone_cell
from training.notebooks.cells.secrets import build_secrets_cell
from training.notebooks.cells.sync import (
    build_continuous_sync_cell,
    build_fuse_mount_cell,
    build_push_cell,
    build_training_cell,
)
from training.utils.paths import get_project_root


class TestNotebookSubsystem(unittest.TestCase):
    """Test suite for modular notebook architecture and baseline parity."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_nb_")
        self.project_root = get_project_root()

        config_path = self.project_root / "config.yaml"
        self.config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        self.registry = load_registry(self.config, self.project_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_cells_base_helpers(self) -> None:
        """Verify that make_code_cell and make_markdown_cell return valid nbformat dicts."""
        code_cell = make_code_cell(["print('hello')\n"])
        self.assertEqual(code_cell["cell_type"], "code")
        self.assertEqual(code_cell["source"], ["print('hello')\n"])
        self.assertIn("metadata", code_cell)
        self.assertIn("outputs", code_cell)
        self.assertIsNone(code_cell["execution_count"])

        md_cell = make_markdown_cell(["# Header\n"])
        self.assertEqual(md_cell["cell_type"], "markdown")
        self.assertEqual(md_cell["source"], ["# Header\n"])
        self.assertIn("metadata", md_cell)
        self.assertNotIn("outputs", md_cell)

    def test_metadata_resolution(self) -> None:
        """Verify metadata extraction for both vision and forex models."""
        mirnet_meta = resolve_model_metadata("mirnet_exposure", self.registry, self.config)
        self.assertIsInstance(mirnet_meta, ModelNotebookMeta)
        self.assertEqual(mirnet_meta.identity.model_key, "mirnet_exposure")
        self.assertFalse(mirnet_meta.identity.is_forex)
        self.assertEqual(mirnet_meta.identity.pascal_name, "MirnetExposure")

        forex_meta = resolve_model_metadata("forex_predictor", self.registry, self.config)
        self.assertTrue(forex_meta.identity.is_forex)
        self.assertEqual(forex_meta.identity.pascal_name, "ForexPredictor")

    def test_env_and_sentinel_cell(self) -> None:
        """Verify runtime env loading and sentinel cell generation."""
        env_vars = _load_runtime_env()
        self.assertIsInstance(env_vars, dict)
        self.assertIn("PYTHONUTF8", env_vars)
        self.assertNotIn("CUDA_FORCE_PTX_JIT", env_vars)

        env_lines = _build_env_var_lines(env_vars)
        self.assertTrue(len(env_lines) > 0)
        self.assertTrue(all(line.startswith("os.environ[") for line in env_lines))

        k_sentinel = build_sentinel_cell("kaggle")
        self.assertEqual(k_sentinel["cell_type"], "code")
        self.assertTrue(any("Kaggle:" in line for line in k_sentinel["source"]))

        c_sentinel = build_sentinel_cell("colab")
        self.assertEqual(c_sentinel["cell_type"], "code")
        self.assertTrue(any("Google Colab:" in line for line in c_sentinel["source"]))

    def _assert_code_cell(self, cell: dict, needle: str | None) -> None:
        """Assert ``cell`` is a code cell whose source mentions ``needle`` (if given)."""
        self.assertEqual(cell["cell_type"], "code")
        if needle is not None:
            self.assertTrue(any(needle in line for line in cell["source"]))

    def test_individual_cell_generators(self) -> None:
        """Verify that individual cell builders construct structurally sound cells."""
        meta = resolve_model_metadata("mirnet_exposure", self.registry, self.config)

        suite_folder = ecosystem().service(TRAINING_SERVICE).project
        # builder output -> a substring one of its source lines must contain (None: any)
        cells = {
            "clone": (build_clone_cell("kaggle"), suite_folder),
            "install": (build_install_cell("kaggle"), "pip"),
            "symlink": (build_symlink_cell(meta, "kaggle"), "LemGendaryDatasets"),
            "hub_prep": (build_hub_prep_cell(meta, "kaggle"), "LemGendaryModels"),
            "recovery": (build_checkpoint_recovery_cell(meta, "kaggle"), "checkpoints"),
            "stealth": (build_stealth_cell(meta, "kaggle"), None),
            "pth": (build_pth_cell(meta), None),
            "onnx_fp32": (build_onnx_fp32_cell(meta), None),
            "onnx_fp16": (build_onnx_fp16_cell(meta), None),
            "secrets": (build_secrets_cell("kaggle"), None),
            "fuse": (build_fuse_mount_cell(), None),
            "sync": (build_continuous_sync_cell(meta), None),
            "train": (build_training_cell(meta, "kaggle"), None),
            "push": (build_push_cell(meta, "kaggle"), None),
        }
        for name, (cell, needle) in cells.items():
            with self.subTest(cell=name):
                self._assert_code_cell(cell, needle)

        # The secrets cells export a KGAT_ token through KAGGLE_API_TOKEN, never kaggle.json.
        for platform in ("kaggle", "colab"):
            source = "".join(build_secrets_cell(platform)["source"])
            self.assertIn("os.environ['KAGGLE_API_TOKEN'] = kaggle_token", source)
            self.assertNotIn("kaggle.json", source)

    def test_backward_compatibility_shim(self) -> None:
        """Verify that training.notebook_generator re-exports all expected functions."""
        self.assertTrue(callable(legacy.generate_inference_notebook))
        self.assertTrue(callable(legacy.generate_usage_notebook))
        self.assertTrue(callable(legacy.generate_training_notebook))
        self.assertTrue(callable(legacy.generate_colab_inference_notebook))
        self.assertTrue(callable(legacy.generate_colab_usage_notebook))
        self.assertTrue(callable(legacy.generate_colab_training_notebook))
        # The private env helpers are part of the shim's declared re-export surface.
        for name in ("_load_runtime_env", "_build_env_var_lines"):
            self.assertIn(name, legacy.__all__)
            self.assertTrue(callable(vars(legacy)[name]))

    def _assert_valid_notebook(self, nb_path: Path) -> None:
        """Assert notebook exists, is valid JSON, and adheres to nbformat 4 schema."""
        self.assertTrue(nb_path.exists(), f"Notebook missing: {nb_path}")
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("nbformat"), 4, f"Invalid nbformat in {nb_path.name}")
        self.assertGreaterEqual(
            data.get("nbformat_minor", 0), 2, f"Invalid nbformat_minor in {nb_path.name}"
        )
        self.assertIn("cells", data, f"Missing cells in {nb_path.name}")
        cells = data["cells"]
        self.assertGreater(
            len(cells), 5, f"Notebook {nb_path.name} has too few cells ({len(cells)})"
        )

        cell_types = [c.get("cell_type") for c in cells]
        self.assertIn("markdown", cell_types, f"No markdown cells in {nb_path.name}")
        self.assertIn("code", cell_types, f"No code cells in {nb_path.name}")
        for idx, cell in enumerate(cells):
            self.assertIn(
                cell.get("cell_type"), ("code", "markdown"), f"Invalid cell type at index {idx}"
            )
            self.assertIn("source", cell, f"Missing source in cell {idx}")
            self.assertIsInstance(cell["source"], list, f"Cell source should be list in cell {idx}")

    def _assert_notebook_parity(self, generated_path: Path, baseline_path: Path) -> None:
        """Assert exact structural equality between generated notebook and baseline fixture."""
        self.assertTrue(generated_path.exists(), f"Generated notebook missing: {generated_path}")
        self.assertTrue(baseline_path.exists(), f"Baseline notebook missing: {baseline_path}")

        with open(generated_path, "r", encoding="utf-8") as f:
            gen_data = json.load(f)
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)

        self.assertEqual(gen_data["nbformat"], base_data["nbformat"])
        self.assertEqual(gen_data["nbformat_minor"], base_data["nbformat_minor"])
        self.assertEqual(gen_data["metadata"], base_data["metadata"])

        gen_cells = gen_data["cells"]
        base_cells = base_data["cells"]
        self.assertEqual(
            len(gen_cells), len(base_cells), f"Cell count mismatch in {generated_path.name}"
        )

        for i, (g_cell, b_cell) in enumerate(zip(gen_cells, base_cells)):
            self.assertEqual(
                g_cell["cell_type"],
                b_cell["cell_type"],
                f"Cell {i} cell_type mismatch in {generated_path.name}",
            )
            self.assertEqual(
                g_cell["source"],
                b_cell["source"],
                f"Cell {i} source mismatch in {generated_path.name}",
            )

    def test_gate_9_mirnet_exposure_parity(self) -> None:
        """Gate 9: Structural verification for mirnet_exposure notebooks."""
        model_key = "mirnet_exposure"
        out_dir = Path(self.temp_dir) / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        generate_inference_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_usage_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_colab_inference_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_colab_usage_notebook(model_key, str(out_dir), self.registry, self.config)

        self._assert_valid_notebook(out_dir / f"{model_key}_kaggle_training.ipynb")
        self._assert_valid_notebook(out_dir / f"{model_key}_kaggle_usage.ipynb")
        self._assert_valid_notebook(out_dir / f"{model_key}_colab_training.ipynb")
        self._assert_valid_notebook(out_dir / f"{model_key}_colab_usage.ipynb")

    def test_gate_9_forex_predictor_parity(self) -> None:
        """Gate 9: Structural verification for forex_predictor notebooks."""
        model_key = "forex_predictor"
        out_dir = Path(self.temp_dir) / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        generate_inference_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_usage_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_colab_inference_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_colab_usage_notebook(model_key, str(out_dir), self.registry, self.config)

        self._assert_valid_notebook(out_dir / f"{model_key}_kaggle_training.ipynb")
        self._assert_valid_notebook(out_dir / f"{model_key}_kaggle_usage.ipynb")
        self._assert_valid_notebook(out_dir / f"{model_key}_colab_training.ipynb")
        self._assert_valid_notebook(out_dir / f"{model_key}_colab_usage.ipynb")

    def test_training_notebook_generators_emit_valid_notebooks(self) -> None:
        """Training notebook generators (used by lemgendary-datasets) emit valid notebooks.

        The delegation itself is covered by the datasets contract test.
        """
        model_key = "mirnet_exposure"
        out_dir = Path(self.temp_dir) / f"{model_key}_ds_ssot"
        out_dir.mkdir(parents=True, exist_ok=True)

        p_train = generate_training_notebook(model_key, str(out_dir), self.registry, self.config)
        p_colab = generate_colab_training_notebook(
            model_key, str(out_dir), self.registry, self.config
        )

        self.assertIsNotNone(p_train)
        self.assertIsNotNone(p_colab)
        assert p_train is not None
        assert p_colab is not None

        self.assertTrue(Path(p_train).exists())
        self.assertTrue(Path(p_colab).exists())
        self._assert_valid_notebook(Path(p_train))
        self._assert_valid_notebook(Path(p_colab))


if __name__ == "__main__":
    unittest.main()

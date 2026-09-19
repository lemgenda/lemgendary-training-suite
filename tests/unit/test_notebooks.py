"""Unit tests for the modular notebook generation subsystem (Phase 9 / Gate 9).

Validates cell builders, registry metadata resolution, backward compatibility,
and exact structural parity against baseline notebooks for mirnet_exposure and forex_predictor.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

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
from training.notebooks.cells.sync import (
    build_continuous_sync_cell,
    build_fuse_mount_cell,
    build_push_cell,
    build_secrets_cell,
    build_training_cell,
)
from training.utils.paths import get_project_root


class TestNotebookSubsystem(unittest.TestCase):
    """Test suite for modular notebook architecture and baseline parity."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_nb_")
        self.project_root = get_project_root()
        self.baseline_root = self.project_root / "tests" / "fixtures" / "baseline_notebooks"

        import yaml
        config_path = self.project_root / "config.yaml"
        self.config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        self.registry = load_registry(self.config, self.project_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

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
        self.assertEqual(mirnet_meta.model_key, "mirnet_exposure")
        self.assertFalse(mirnet_meta.is_forex)
        self.assertEqual(mirnet_meta.pascal_name, "MirnetExposure")

        forex_meta = resolve_model_metadata("forex_predictor", self.registry, self.config)
        self.assertTrue(forex_meta.is_forex)
        self.assertEqual(forex_meta.pascal_name, "ForexPredictor")

    def test_env_and_sentinel_cell(self) -> None:
        """Verify runtime env loading and sentinel cell generation."""
        env_vars = _load_runtime_env()
        self.assertIsInstance(env_vars, dict)
        self.assertIn("PYTHONUTF8", env_vars)
        self.assertNotIn("CUDA_FORCE_PTX_JIT", env_vars)

        env_lines = _build_env_var_lines(env_vars)
        self.assertTrue(len(env_lines) > 0)
        self.assertTrue(all(l.startswith("os.environ[") for l in env_lines))

        k_sentinel = build_sentinel_cell("kaggle")
        self.assertEqual(k_sentinel["cell_type"], "code")
        self.assertTrue(any("Kaggle:" in line for line in k_sentinel["source"]))

        c_sentinel = build_sentinel_cell("colab")
        self.assertEqual(c_sentinel["cell_type"], "code")
        self.assertTrue(any("Google Colab:" in line for line in c_sentinel["source"]))

    def test_individual_cell_generators(self) -> None:
        """Verify that individual cell builders construct structurally sound cells."""
        meta = resolve_model_metadata("mirnet_exposure", self.registry, self.config)

        clone_cell = build_clone_cell("kaggle")
        self.assertEqual(clone_cell["cell_type"], "code")
        self.assertTrue(any("lemgendary-training-suite" in l for l in clone_cell["source"]))

        install_cell = build_install_cell("kaggle")
        self.assertEqual(install_cell["cell_type"], "code")
        self.assertTrue(any("pip" in l for l in install_cell["source"]))

        symlink_cell = build_symlink_cell(meta, "kaggle")
        self.assertEqual(symlink_cell["cell_type"], "code")
        self.assertTrue(any("LemGendaryDatasets" in l for l in symlink_cell["source"]))

        hub_cell = build_hub_prep_cell(meta, "kaggle")
        self.assertEqual(hub_cell["cell_type"], "code")
        self.assertTrue(any("LemGendaryModels" in l for l in hub_cell["source"]))

        recovery_cell = build_checkpoint_recovery_cell(meta, "kaggle")
        self.assertEqual(recovery_cell["cell_type"], "code")
        self.assertTrue(any("checkpoints" in l for l in recovery_cell["source"]))

        stealth_cell = build_stealth_cell(meta, "kaggle")
        self.assertEqual(stealth_cell["cell_type"], "code")

        pth_cell = build_pth_cell(meta)
        self.assertEqual(pth_cell["cell_type"], "code")

        onnx32_cell = build_onnx_fp32_cell(meta)
        self.assertEqual(onnx32_cell["cell_type"], "code")

        onnx16_cell = build_onnx_fp16_cell(meta)
        self.assertEqual(onnx16_cell["cell_type"], "code")

        secrets_cell = build_secrets_cell("kaggle")
        self.assertEqual(secrets_cell["cell_type"], "code")

        fuse_cell = build_fuse_mount_cell()
        self.assertEqual(fuse_cell["cell_type"], "code")

        sync_cell = build_continuous_sync_cell(meta)
        self.assertEqual(sync_cell["cell_type"], "code")

        train_cell = build_training_cell(meta, "kaggle")
        self.assertEqual(train_cell["cell_type"], "code")

        push_cell = build_push_cell(meta, "kaggle")
        self.assertEqual(push_cell["cell_type"], "code")

    def test_backward_compatibility_shim(self) -> None:
        """Verify that training.notebook_generator re-exports all expected functions."""
        import training.notebook_generator as legacy

        self.assertTrue(callable(legacy.generate_inference_notebook))
        self.assertTrue(callable(legacy.generate_usage_notebook))
        self.assertTrue(callable(legacy.generate_training_notebook))
        self.assertTrue(callable(legacy.generate_colab_inference_notebook))
        self.assertTrue(callable(legacy.generate_colab_usage_notebook))
        self.assertTrue(callable(legacy.generate_colab_training_notebook))
        self.assertTrue(callable(legacy._load_runtime_env))
        self.assertTrue(callable(legacy._build_env_var_lines))

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
        self.assertEqual(len(gen_cells), len(base_cells), f"Cell count mismatch in {generated_path.name}")

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
        """Gate 9: Structural parity verification for mirnet_exposure notebooks."""
        model_key = "mirnet_exposure"
        out_dir = Path(self.temp_dir) / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        generate_inference_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_usage_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_colab_inference_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_colab_usage_notebook(model_key, str(out_dir), self.registry, self.config)

        base_dir = self.baseline_root / model_key
        if not base_dir.exists():
            self.skipTest("Baseline notebooks not found on disk")

        self._assert_notebook_parity(out_dir / f"{model_key}_training.ipynb", base_dir / f"{model_key}_training.ipynb")
        self._assert_notebook_parity(out_dir / f"{model_key}-usage.ipynb", base_dir / f"{model_key}-usage.ipynb")
        self._assert_notebook_parity(out_dir / f"{model_key}_colab_training.ipynb", base_dir / f"{model_key}_colab_training.ipynb")
        self._assert_notebook_parity(out_dir / f"{model_key}-colab-usage.ipynb", base_dir / f"{model_key}-colab-usage.ipynb")

    def test_gate_9_forex_predictor_parity(self) -> None:
        """Gate 9: Structural parity verification for forex_predictor notebooks."""
        model_key = "forex_predictor"
        out_dir = Path(self.temp_dir) / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        generate_inference_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_usage_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_colab_inference_notebook(model_key, str(out_dir), self.registry, self.config)
        generate_colab_usage_notebook(model_key, str(out_dir), self.registry, self.config)

        base_dir = self.baseline_root / model_key
        if not base_dir.exists():
            self.skipTest("Baseline notebooks not found on disk")

        self._assert_notebook_parity(out_dir / f"{model_key}_training.ipynb", base_dir / f"{model_key}_training.ipynb")
        self._assert_notebook_parity(out_dir / f"{model_key}-usage.ipynb", base_dir / f"{model_key}-usage.ipynb")
        self._assert_notebook_parity(out_dir / f"{model_key}_colab_training.ipynb", base_dir / f"{model_key}_colab_training.ipynb")
        self._assert_notebook_parity(out_dir / f"{model_key}-colab-usage.ipynb", base_dir / f"{model_key}-colab-usage.ipynb")

    def test_dataset_compiler_notebook_generator_parity(self) -> None:
        """Verify that lemgendary-datasets tools/notebook_generator matches training suite outputs."""
        ds_repo = Path(__file__).resolve().parent.parent.parent.parent / "lemgendary-datasets"
        ds_tools = ds_repo / "tools"
        if not (ds_tools / "notebook_generator.py").exists():
            self.skipTest("lemgendary-datasets not found in workspace")

        paths_inserted: list[str] = []
        if str(ds_repo) not in sys.path:
            sys.path.insert(0, str(ds_repo))
            paths_inserted.append(str(ds_repo))
        if str(ds_tools) not in sys.path:
            sys.path.insert(0, str(ds_tools))
            paths_inserted.append(str(ds_tools))

        import tools
        ds_tools_str = str(ds_tools)
        if ds_tools_str not in tools.__path__:
            tools.__path__.insert(0, ds_tools_str)

        webdataset_mocked = False
        if "webdataset" not in sys.modules:
            import types
            wds = types.ModuleType("webdataset")
            writer_mod = types.ModuleType("webdataset.writer")
            setattr(writer_mod, "ShardWriter", object)
            setattr(wds, "writer", writer_mod)
            sys.modules["webdataset"] = wds
            sys.modules["webdataset.writer"] = writer_mod
            webdataset_mocked = True

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("ds_notebook_generator", str(ds_tools / "notebook_generator.py"))
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)  # type assertion for mypy/runtime
            ds_nb = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ds_nb)

            # Verify exported symbols
            for sym in [
                "_load_runtime_env",
                "_build_env_var_lines",
                "generate_inference_notebook",
                "generate_usage_notebook",
                "generate_training_notebook",
                "generate_colab_inference_notebook",
                "generate_colab_usage_notebook",
                "generate_colab_training_notebook",
            ]:
                self.assertTrue(hasattr(ds_nb, sym), f"Missing exported symbol in ds_nb: {sym}")

            # Verify generation parity
            model_key = "mirnet_exposure"
            out_dir = Path(self.temp_dir) / f"{model_key}_from_datasets"
            out_dir.mkdir(parents=True, exist_ok=True)

            ds_nb.generate_inference_notebook(model_key, str(out_dir), self.registry, self.config)
            ds_nb.generate_usage_notebook(model_key, str(out_dir), self.registry, self.config)
            ds_nb.generate_colab_inference_notebook(model_key, str(out_dir), self.registry, self.config)
            ds_nb.generate_colab_usage_notebook(model_key, str(out_dir), self.registry, self.config)

            base_dir = self.baseline_root / model_key
            if base_dir.exists():
                self._assert_notebook_parity(out_dir / f"{model_key}_training.ipynb", base_dir / f"{model_key}_training.ipynb")
                self._assert_notebook_parity(out_dir / f"{model_key}-usage.ipynb", base_dir / f"{model_key}-usage.ipynb")
                self._assert_notebook_parity(out_dir / f"{model_key}_colab_training.ipynb", base_dir / f"{model_key}_colab_training.ipynb")
                self._assert_notebook_parity(out_dir / f"{model_key}-colab-usage.ipynb", base_dir / f"{model_key}-colab-usage.ipynb")
        finally:
            if webdataset_mocked:
                if "webdataset.writer" in sys.modules:
                    del sys.modules["webdataset.writer"]
                if "webdataset" in sys.modules:
                    del sys.modules["webdataset"]
            for p in paths_inserted:
                if p in sys.path:
                    sys.path.remove(p)
            if ds_tools_str in tools.__path__:
                tools.__path__.remove(ds_tools_str)


if __name__ == "__main__":
    unittest.main()


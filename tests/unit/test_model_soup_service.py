"""Torch-free tests for the Pareto Model Soup service, API route and CLI wiring (TS D-02)."""

from __future__ import annotations

import ast
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from training import model_soup
from training.model_soup import (
    SoupCheckpointsMissing,
    SoupInvalidRequest,
    SoupRecipe,
    blend_model_soup,
    resolve_models_root,
    resolve_vault_checkpoints,
)
from training.server.routes import soup as soup_route

MODEL = "film_restoration"


class _SoupTree(unittest.TestCase):
    """Temp workspace: <tmp>/suite (project root) and sibling <tmp>/LemGendaryModels."""

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.workspace)
        self.project_root = self.workspace / "suite"
        self.project_root.mkdir()
        (self.project_root / "config.yaml").write_text(
            'paths:\n  checkpoints_root: "../LemGendaryModels"\n', encoding="utf-8"
        )
        self.ckpt_dir = self.workspace / "LemGendaryModels" / MODEL / "checkpoints"
        self.ckpt_dir.mkdir(parents=True)

    def add_vaults(self, *metrics: str) -> None:
        """Create empty vault checkpoints for the given metric names."""
        for m in metrics:
            (self.ckpt_dir / f"vault_{m}.pth").write_bytes(b"x")


class TestSoupPathResolution(_SoupTree):
    """Tests for models-root and vault checkpoint path resolution."""

    def test_models_root_comes_from_config(self) -> None:
        """Verify the models root comes from config.yaml checkpoints_root."""
        (self.project_root / "config.yaml").write_text(
            'paths:\n  checkpoints_root: "../OtherModels"\n', encoding="utf-8"
        )
        self.assertEqual(resolve_models_root(self.project_root), self.workspace / "OtherModels")

    def test_models_root_defaults_to_sibling_repo_without_config(self) -> None:
        """Verify the models root defaults to the sibling LemGendaryModels repo."""
        (self.project_root / "config.yaml").unlink()
        self.assertEqual(
            resolve_models_root(self.project_root), self.workspace / "LemGendaryModels"
        )

    def test_absolute_configured_root_is_rejected(self) -> None:
        """Verify an absolute configured checkpoints_root is rejected."""
        (self.project_root / "config.yaml").write_text(
            f'paths:\n  checkpoints_root: "{self.workspace.as_posix()}"\n', encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            resolve_models_root(self.project_root)

    def test_both_vaults_present_returns_two_files(self) -> None:
        """Verify both requested vaults resolve in order."""
        self.add_vaults("psnr", "lpips")
        paths = resolve_vault_checkpoints(MODEL, self.project_root, ["psnr", "lpips"])
        self.assertEqual([p.name for p in paths], ["vault_psnr.pth", "vault_lpips.pth"])
        self.assertTrue(all(p.is_file() for p in paths))

    def test_default_discovers_all_vaults(self) -> None:
        """Verify default discovery returns all vault_* checkpoints only."""
        self.add_vaults("psnr", "lpips", "ssim")
        (self.ckpt_dir / "best.pth").write_bytes(b"x")
        paths = resolve_vault_checkpoints(MODEL, self.project_root)
        self.assertEqual(
            [p.name for p in paths], ["vault_lpips.pth", "vault_psnr.pth", "vault_ssim.pth"]
        )

    def test_missing_requested_vault_lists_it(self) -> None:
        """Verify a missing requested vault is listed in the error."""
        self.add_vaults("psnr")
        with self.assertRaises(SoupCheckpointsMissing) as ctx:
            resolve_vault_checkpoints(MODEL, self.project_root, ["psnr", "lpips"])
        self.assertEqual(ctx.exception.missing, ["vault_lpips.pth"])
        self.assertIn("vault_lpips.pth", str(ctx.exception))

    def test_missing_model_folder(self) -> None:
        """Verify an unknown model folder raises SoupCheckpointsMissing."""
        with self.assertRaises(SoupCheckpointsMissing):
            resolve_vault_checkpoints("unknown_model", self.project_root)

    def test_no_vaults_is_missing(self) -> None:
        """Verify having no vaults reports the default ones as missing."""
        with self.assertRaises(SoupCheckpointsMissing) as ctx:
            resolve_vault_checkpoints(MODEL, self.project_root)
        self.assertIn("vault_psnr.pth", ctx.exception.missing)

    def test_single_vault_is_invalid(self) -> None:
        """Verify a single vault is an invalid soup request."""
        self.add_vaults("psnr")
        with self.assertRaises(SoupInvalidRequest):
            resolve_vault_checkpoints(MODEL, self.project_root)
        with self.assertRaises(SoupInvalidRequest):
            resolve_vault_checkpoints(MODEL, self.project_root, ["psnr"])

    def test_unsafe_names_are_rejected(self) -> None:
        """Verify unsafe model and donor names are rejected."""
        with self.assertRaises(SoupInvalidRequest):
            resolve_vault_checkpoints("..", self.project_root)
        self.add_vaults("psnr", "lpips")
        with self.assertRaises(SoupInvalidRequest):
            resolve_vault_checkpoints(MODEL, self.project_root, ["psnr", "../lpips"])


class TestBlendService(_SoupTree):
    """Tests for blend_model_soup orchestration."""

    def test_passes_file_list_to_soup_function(self) -> None:
        """Verify the resolved file list is passed to the soup function."""
        self.add_vaults("psnr", "lpips")
        fn = mock.Mock(return_value={})
        result = blend_model_soup(
            MODEL,
            self.project_root,
            SoupRecipe(["psnr", "lpips"], [0.6, 0.4]),
            soup_function=fn,
        )
        args, kwargs = fn.call_args
        self.assertEqual(
            args[0], [self.ckpt_dir / "vault_psnr.pth", self.ckpt_dir / "vault_lpips.pth"]
        )
        self.assertEqual(kwargs["weights"], [0.6, 0.4])
        self.assertEqual(kwargs["output_path"], self.ckpt_dir / "best_soup.pth")
        self.assertEqual(result.output_path, self.ckpt_dir / "best_soup.pth")

    def test_bare_output_name_lands_in_checkpoints(self) -> None:
        """Verify a bare output name is written into the checkpoints directory."""
        self.add_vaults("psnr", "lpips")
        result = blend_model_soup(
            MODEL, self.project_root, SoupRecipe(output="mix.pth"), soup_function=mock.Mock()
        )
        self.assertEqual(result.output_path, self.ckpt_dir / "mix.pth")

    def test_weight_count_mismatch_is_invalid(self) -> None:
        """Verify a weight/donor count mismatch is an invalid request."""
        self.add_vaults("psnr", "lpips")
        fn = mock.Mock()
        with self.assertRaises(SoupInvalidRequest):
            blend_model_soup(MODEL, self.project_root, SoupRecipe(weights=[1.0]), soup_function=fn)
        fn.assert_not_called()


class TestSoupRoute(_SoupTree):
    """Tests for the soup REST route."""

    def setUp(self) -> None:
        super().setUp()
        app = FastAPI()
        app.include_router(soup_route.router, prefix="/api")
        app.state.server_state = SimpleNamespace(project_root=self.project_root)
        self.client = TestClient(app)
        self.soup_fn = mock.Mock(return_value={})
        patcher = mock.patch.object(model_soup, "_default_soup_function", return_value=self.soup_fn)
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, body: dict[str, Any] | None = None) -> Any:
        """POST to the soup route, with ``body`` as JSON when given."""
        url = f"/api/models/{MODEL}/soup"
        return self.client.post(url, json=body) if body is not None else self.client.post(url)

    def test_empty_body_blends_all_vaults(self) -> None:
        """Verify an empty body blends all vaults."""
        self.add_vaults("psnr", "lpips")
        res = self.post()
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model_key"], MODEL)
        self.assertEqual(Path(data["soup_checkpoint"]), self.ckpt_dir / "best_soup.pth")
        self.assertEqual(data["donors"], ["vault_lpips.pth", "vault_psnr.pth"])
        self.assertEqual(len(self.soup_fn.call_args.args[0]), 2)

    def test_explicit_donors_and_weights(self) -> None:
        """Verify explicit donors and weights are forwarded."""
        self.add_vaults("psnr", "lpips", "ssim")
        res = self.post({"donors": ["psnr", "lpips"], "weights": [0.7, 0.3]})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["donors"], ["vault_psnr.pth", "vault_lpips.pth"])
        self.assertEqual(self.soup_fn.call_args.kwargs["weights"], [0.7, 0.3])

    def test_missing_checkpoints_return_404(self) -> None:
        """Verify missing checkpoints return 404."""
        self.add_vaults("psnr")
        res = self.post({"donors": ["psnr", "lpips"]})
        self.assertEqual(res.status_code, 404)
        self.assertIn("vault_lpips.pth", res.json()["detail"])
        self.soup_fn.assert_not_called()

    def test_unknown_model_returns_404(self) -> None:
        """Verify an unknown model returns 404."""
        res = self.client.post("/api/models/unknown_model/soup")
        self.assertEqual(res.status_code, 404)

    def test_single_checkpoint_returns_422(self) -> None:
        """Verify a single checkpoint returns 422."""
        self.add_vaults("psnr")
        res = self.post()
        self.assertEqual(res.status_code, 422)
        self.soup_fn.assert_not_called()


class TestCliUsesSharedService(unittest.TestCase):
    """lemtrain imports torch-backed services, so check the soup command's source instead."""

    def test_soup_command_delegates_to_service(self) -> None:
        """Verify the CLI soup command delegates to the shared service."""
        src_path = Path(model_soup.__file__).resolve().parent / "cli" / "lemtrain.py"
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "soup")
        imports = {
            (n.module, a.name) for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names
        }
        self.assertIn(("training", "model_soup"), imports)
        calls = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
        self.assertIn("model_soup.blend_model_soup", calls)
        self.assertNotIn("create_pareto_model_soup", calls)
        self.assertNotIn("LemGendaryModels", ast.unparse(fn))

    def test_route_delegates_to_service(self) -> None:
        """Verify the REST route uses the shared blend service."""
        self.assertIs(soup_route.blend_model_soup, blend_model_soup)


if __name__ == "__main__":
    unittest.main()

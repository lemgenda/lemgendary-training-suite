"""Regression tests for Prompt 14 training suite specification audit fixes."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from typer.testing import CliRunner

from models.factory import _merge_llm_entry_kwargs
from tools.train_all import _approve, _has_artifacts
from training.cli.lemtrain import app
from training.core_loop import build_training_paths, clean_previous_run
from training.doc_generator import read_model_metrics
from training.governance.governor_setup import build_curriculum
from training.governance.logit_cap_sync import sync_criterion_actuators
from training.registry.preprocessing import task_type_of
from training.server.routes.gui_model_stats import _identity_fields


def test_train_all_has_artifacts_detects_canonical_best_pth(tmp_path: Path) -> None:
    """``_has_artifacts`` detects canonical ``checkpoints/best.pth`` in the model hub folder."""
    ckpt_dir = tmp_path / "LemGendaryModels" / "mirnet_exposure" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "best.pth").write_bytes(b"ckpt")
    args = argparse.Namespace(force=False, epochs=10)
    with mock.patch("tools.train_all.get_model_hub_dir", return_value=ckpt_dir.parent):
        assert _has_artifacts(tmp_path, "mirnet_exposure", args) is True


def test_train_all_approve_does_not_double_read_stdin() -> None:
    """``_approve`` reads ``input()`` once and sets ``auto_accept`` when ``all`` is entered."""
    with mock.patch("builtins.input", return_value="all") as mocked_input:
        approved, auto_accept = _approve("mirnet_exposure", auto_accept=False)
    assert mocked_input.call_count == 1
    assert approved is True
    assert auto_accept is True

    with mock.patch("builtins.input", return_value="y"):
        approved_y, auto_accept_y = _approve("mirnet_exposure", auto_accept=False)
    assert approved_y is True
    assert auto_accept_y is False


def test_factory_merges_llm_curriculum_kwargs() -> None:
    """Top-level ``base_foundation_model`` and ``curriculum`` LoRA keys merge into LLM kwargs."""
    entry = {
        "category": "llm",
        "class_name": "AIHelperLemGend",
        "base_foundation_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "curriculum": {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "ce_chunk_size": 256,
        },
    }
    merged = _merge_llm_entry_kwargs(entry, {"lora_r": 8})
    assert merged["base_model_name_or_path"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert merged["lora_r"] == 8
    assert merged["lora_alpha"] == 32
    assert merged["ce_chunk_size"] == 256


def test_clean_previous_run_removes_canonical_and_vault_checkpoints(tmp_path: Path) -> None:
    """``clean_previous_run`` removes ``latest.pth``, ``vault_*.pth``, SWA, and curriculum files."""
    paths = build_training_paths(tmp_path, "mirnet_exposure")
    assert paths.models_hub is not None
    ckpt_dir = paths.models_hub.checkpoint_dir
    hub_dir = paths.models_hub.root
    for name in (
        "progress.pth",
        "latest.pth",
        "best.pth",
        "swa_state.pth",
        "curriculum_state.json",
        "vault_psnr.pth",
        "vault_swa.pth",
    ):
        (ckpt_dir / name).write_bytes(b"x")
    (hub_dir / "metrics.csv").write_text("Epoch\n1\n", encoding="utf-8")

    clean_previous_run(paths, "mirnet_exposure")

    for name in (
        "progress.pth",
        "latest.pth",
        "best.pth",
        "swa_state.pth",
        "curriculum_state.json",
        "vault_psnr.pth",
        "vault_swa.pth",
    ):
        assert not (ckpt_dir / name).exists(), name
    assert not (hub_dir / "metrics.csv").exists()


def test_sync_criterion_actuators_propagates_governor_state() -> None:
    """``sync_criterion_actuators`` copies governor stabilizer weights into ``criterion.stab``."""
    forex_dual = SimpleNamespace(stab={})
    criterion = SimpleNamespace(
        stab={"softmax_temp": 1.0, "rank_weight": 0.8},
        forex_dual=forex_dual,
    )
    governor = SimpleNamespace(
        current_temp=0.75,
        current_clamp=20.0,
        current_rank_weight=1.25,
        current_rank_margin=0.08,
        current_spearman_weight=0.6,
        current_emd_weight=1.5,
        current_ssim_weight=0.1,
        current_lpips_weight=0.04,
    )
    assert sync_criterion_actuators(criterion, governor) is True
    assert criterion.stab["softmax_temp"] == pytest.approx(0.75)
    assert criterion.stab["logit_clamp"] == pytest.approx(20.0)
    assert criterion.stab["rank_weight"] == pytest.approx(1.25)
    assert criterion.stab["lpips_weight"] == pytest.approx(0.04)
    assert forex_dual.stab is criterion.stab
    assert sync_criterion_actuators(criterion, governor) is False


def test_task_type_ssot_in_preprocessing_and_gui_stats() -> None:
    """``task_type_of`` and ``_identity_fields`` use ``task_type -> dataset_type -> category``."""
    info = {"task_type": "quality", "dataset_type": "restoration", "category": "vision"}
    assert task_type_of(info) == "quality"
    assert _identity_fields("m", info)["task_type"] == "quality"
    face_info = {"dataset_type": "face", "category": "detection"}
    assert _identity_fields("m", face_info)["task_type"] == "face"
    assert _identity_fields("m", {})["task_type"] == "general"


def test_llm_curriculum_res_ladder_is_single_context_rung() -> None:
    """An LLM without explicit ``res_ladder`` keeps a single-rung sequence length ladder."""
    cur = build_curriculum({"category": "llm", "input_size": 512}, {}, "text_instruction")
    assert cur.res_ladder == [512]


def test_doc_generator_reads_one_indexed_epochs_without_off_by_one(tmp_path: Path) -> None:
    """``read_model_metrics`` returns ``int(last_ep)`` for 1-indexed ``Epoch`` rows."""
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("Epoch,Train_Loss,Val_Loss\n1,0.5,0.4\n20,0.2,0.1\n", encoding="utf-8")
    epochs_trained, _metrics = read_model_metrics(str(csv_path), "mirnet_exposure")
    assert epochs_trained == 20


def test_cli_notebooks_generate_subcommand() -> None:
    """``lemtrain notebooks generate`` and ``lemtrain notebooks list`` both dispatch."""
    runner = CliRunner()
    with mock.patch("training.cli.lemtrain.NotebookService") as service:
        service.return_value.generate_all_models.return_value = {
            "mirnet_exposure": {"train": Path("train.ipynb")}
        }
        res_all = runner.invoke(app, ["notebooks", "generate", "--all"])
        assert res_all.exit_code == 0, res_all.output
        service.return_value.generate_all_models.assert_called_once()

        service.return_value.generate_notebooks.return_value = {"train": Path("train.ipynb")}
        res_one = runner.invoke(
            app, ["notebooks", "generate", "--model", "mirnet_exposure", "-p", "colab"]
        )
        assert res_one.exit_code == 0, res_one.output
        assert (
            service.return_value.generate_notebooks.call_args.kwargs["model_key"]
            == "mirnet_exposure"
        )
        assert service.return_value.generate_notebooks.call_args.kwargs["platform"] == "colab"

"""lemtrain llm: causal LLM instruction tuning and tri-format export."""

from __future__ import annotations

from typing import Annotated

import typer

from training.cli.service_calls import announce_sidecar, export_targets, run_training

# LLM Operations Sub-Commands
llm_app = typer.Typer(
    name="llm", help="Causal LLM instruction tuning and tri-format export operations."
)


@llm_app.command("train")
def llm_train(  # pylint: disable=too-many-arguments  # one parameter per CLI option
    model: str = typer.Argument(
        "ai_helper_lemgend", help="LLM model key (e.g. ai_helper_lemgend, test_llm)."
    ),
    *,
    epochs: int | None = typer.Option(None, "--epochs", "-e", help="Total training epochs."),
    batch_size: int | None = typer.Option(
        None, "--batch-size", "-b", help="Per-device micro-batch size."
    ),
    lr: float | None = typer.Option(None, "--lr", help="Learning rate for LoRA adapters."),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help="Canonical preset profile from presets.yaml."
    ),
    env: str = typer.Option(
        "local", "--env", help="Execution environment ('local', 'kaggle', 'colab')."
    ),
    clean: bool = typer.Option(False, "--clean", help="Wipe local checkpoints and start fresh."),
    auto_sync: bool = typer.Option(
        False, "--auto-sync", help="Enable cloud checkpoint synchronization."
    ),
) -> None:
    """Launch instruction fine-tuning for any causal LLM architecture."""
    announce_sidecar()
    typer.echo(f"[START] Initiating LLM training run for model '{model}' (env={env})...")
    summary = run_training(
        model,
        env=env,
        preset=preset,
        lr=lr,
        epochs=epochs,
        clean=clean,
        batch_size=batch_size,
        auto_sync=auto_sync,
    )
    typer.echo(f"[COMPLETE] Final training status: {summary.status} at epoch {summary.final_epoch}")


@llm_app.command("export")
def llm_export(
    model: str = typer.Argument("ai_helper_lemgend", help="LLM model key to export."),
    checkpoint: str | None = typer.Option(
        None, "--checkpoint", "-c", help="Path to checkpoint file."
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Target directory for binaries."
    ),
    target: Annotated[
        list[str] | None,
        typer.Option(
            "--target",
            "-t",
            help="Target formats ('qlora_adapter', 'merged_safetensors', 'gguf_fp16').",
        ),
    ] = None,
) -> None:
    """Compile and export tri-format binaries for an LLM."""
    typer.echo(f"[EXPORT] Compiling tri-format LLM export targets for '{model}'...")
    results = export_targets(model, checkpoint, output_dir, target)
    typer.echo("[SUCCESS] LLM Artifacts exported:")
    for tgt, path in results.items():
        typer.echo(f"  [{tgt}] -> {path}")

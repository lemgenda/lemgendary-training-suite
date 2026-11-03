"""lemtrain llm: causal LLM instruction tuning and tri-format export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Annotated

import typer

from training.cli.option_groups import option_group
from training.cli.service_calls import (
    RunFlags,
    announce_sidecar,
    export_targets,
    run_training,
)

# LLM Operations Sub-Commands
llm_app = typer.Typer(
    name="llm", help="Causal LLM instruction tuning and tri-format export operations."
)


@dataclass(frozen=True)
class LLMHyper:
    """Hyper-parameter options of ``lemtrain llm train``."""

    epochs: Annotated[int | None, typer.Option("--epochs", "-e", help="Total training epochs.")] = (
        None
    )
    batch_size: Annotated[
        int | None, typer.Option("--batch-size", "-b", help="Per-device micro-batch size.")
    ] = None
    lr: Annotated[float | None, typer.Option("--lr", help="Learning rate for LoRA adapters.")] = (
        None
    )


@llm_app.command("train")
@option_group("hyper", LLMHyper)
@option_group("run", RunFlags)
def llm_train(
    model: str = typer.Argument(
        "ai_helper_lemgend", help="LLM model key (e.g. ai_helper_lemgend, test_llm)."
    ),
    *,
    hyper: LLMHyper,
    run: RunFlags,
) -> None:
    """Launch instruction fine-tuning for any causal LLM architecture."""
    announce_sidecar()
    typer.echo(f"[START] Initiating LLM training run for model '{model}' (env={run.env})...")
    summary = run_training(model, **asdict(hyper), **asdict(run))
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

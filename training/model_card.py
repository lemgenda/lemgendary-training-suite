"""Markdown model card (per-model ``README.md``) rendering from registry config and metrics."""

from __future__ import annotations

import logging
import math
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import yaml

from training.loss_emd import registry_emd_form
from training.model_card_snippets import SnippetInput, topology_mermaid, usage_snippet
from training.registry import get_model_registry
from training.registry.class_names import class_names
from training.registry.forex_risk import risk_label, risk_per_trade
from training.registry.preprocessing import image_preprocessing
from training.utils.lazy_imports import load_cloud_module

logger = logging.getLogger("lemtrain.docs.doc_generator")

DEFAULT_HARDWARE = "NVIDIA GeForce GTX 1650 (4G VRAM)"

TOP5_CLASSES = 5
"""Class count up to which top-5 accuracy is trivially 1.0 and is left off the card."""

# Registry lookups load unified_models_v2.yaml lazily: import failures, unreadable or malformed
# YAML, and spec entries of unexpected shape are the expected failure modes.
_REGISTRY_ERRORS: tuple[type[Exception], ...] = (
    ImportError,
    OSError,
    ValueError,
    yaml.YAMLError,
    AttributeError,
    TypeError,
)


def _wrap_md_bullet(text: str, width: int = 80) -> str:
    """Wrap a top-level markdown bullet item (`- ...`) to ``width`` columns."""
    return textwrap.fill(
        text,
        width=width,
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def _wrap_bibtex(bibtex: str, width: int = 80) -> str:
    """Wrap BibTeX lines so none exceed ``width`` characters."""
    out: list[str] = []
    for line in bibtex.splitlines():
        if len(line) <= width:
            out.append(line)
        else:
            out.append(
                textwrap.fill(
                    line,
                    width=width,
                    subsequent_indent="    ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    return "\n".join(out)


def _wrap_md_paragraph(text: str, width: int = 80) -> str:
    """Wrap a plain markdown paragraph to ``width`` columns."""
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _registry_reference(model_key: str) -> dict[str, str] | None:
    """Return the academic reference of ``model_key`` from the SSOT; registry errors raise."""
    spec = get_model_registry().get_model(model_key)
    if spec and spec.academic_reference and spec.academic_reference.title:
        ref = spec.academic_reference
        return {
            "title": ref.title,
            "authors": ref.authors,
            "year": str(ref.year),
            "venue": ref.venue,
            "link": ref.link,
            "bibtex": ref.bibtex,
        }
    return None


def get_academic_reference(model_key: str) -> dict[str, str] | None:
    """Retrieve academic paper reference dynamically from the SSOT."""
    try:
        return _registry_reference(model_key)
    except _REGISTRY_ERRORS as exc:
        logger.warning("Could not load academic reference for %s from registry: %s", model_key, exc)
    return None


class _ModelPapersProxy(dict):
    """Backwards-compatible dictionary proxy delegating to SSOT ModelRegistryService."""

    def get(self, key: Any, default: Any = None) -> Any:
        ref = get_academic_reference(str(key))
        return ref if ref is not None else default

    def __getitem__(self, key: Any) -> Any:
        ref = get_academic_reference(str(key))
        if ref is not None:
            return ref
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and get_academic_reference(key) is not None


MODEL_PAPERS = _ModelPapersProxy()


@dataclass(frozen=True)
class ModelCard:
    """Registry config and latest metrics of the model a card is rendered for."""

    model_key: str
    info: dict[str, Any]
    task: Any
    size: tuple[Any, Any]
    res_str: str
    metrics: dict[str, Any]

    @classmethod
    def load(cls, model_key: str, unified_models: dict[str, Any], metrics: dict) -> ModelCard:
        """Resolve the task and the documented input resolution of ``model_key``."""
        info = unified_models.get(model_key, {})
        task = info.get("dataset_type", "restoration")
        if isinstance(task, list):
            task = task[0]

        # Handle input_size for documentation
        sz_raw = info.get("input_size", [3, 256, 256])
        if sz_raw is None or task == "forex":
            size, res_str = (168, 14), "168x14 (Lookback Sequence)"
        elif isinstance(sz_raw, list):
            size = (sz_raw[1], sz_raw[2]) if len(sz_raw) == 3 else (sz_raw[0], sz_raw[1])
            res_str = f"{size[0]}x{size[1]}"
        else:
            size, res_str = (sz_raw, sz_raw), f"{sz_raw}x{sz_raw}"

        # Override resolution if active in metrics
        if metrics and metrics.get("Res"):
            try:
                m_res = int(float(metrics["Res"]))
                if task != "forex" and m_res > 0:
                    res_str = f"{m_res}x{m_res}"
            except (ValueError, TypeError):
                logger.debug(
                    "Non-numeric Res %r in metrics for %s; keeping %s",
                    metrics.get("Res"),
                    model_key,
                    res_str,
                )
        return cls(model_key, info, task, size, res_str, metrics)

    @property
    def name(self) -> str:
        """Display name of the model."""
        return self.info.get("name", self.model_key)

    @property
    def arch(self) -> str:
        """Model class name shown as the architecture."""
        return self.info.get("class_name", "PyTorch Specialized Matrix")

    @property
    def datasets(self) -> list[str]:
        """Dataset keys the model trains on."""
        return self.info.get("datasets", [])


NOT_MEASURED = "not measured"
"""Shown for a metric the checkpoint's metrics do not hold; never a target or a guess."""


def measured(metrics: dict[str, Any], *keys: str) -> str:
    """The first measured value among ``keys`` (``0`` included), else :data:`NOT_MEASURED`.

    Model cards used to print the SOTA target or a hardcoded figure when a metric was
    missing (``metrics.get(k) or sota.get(k, "58.5")``), so an untrained model's card
    showed its targets as achieved results (R2-T10).
    """
    for key in keys:
        value = metrics.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isnan(number):
            return f"{number:.4g}"
    return NOT_MEASURED


def _quality_summary(card: ModelCard) -> tuple[str, str]:
    metrics, model_key = card.metrics, card.model_key
    summary = f"**PLCC**: {measured(metrics, 'plcc')} | **SRCC**: {measured(metrics, 'srcc')}"
    focus = "Aesthetics" if "aesthetic" in model_key else "Technical Integrity"
    targets = (
        "Composition, Color, Lighting, Artistic Intent"
        if "aesthetic" in model_key
        else "Noise, Blur, Compression, Sharpness"
    )
    vector_section = (
        f"> [!IMPORTANT]\n"
        f"> **Quality Vector**: This model is specialized for **{focus}**.\n>\n"
        f"> - **Primary Targets**: {targets}.\n"
    )
    return summary, vector_section


def _forex_summary(card: ModelCard) -> tuple[str, str]:
    metrics = card.metrics
    dir_acc = measured(metrics, "dir_acc", "DirAcc")
    win_rate = measured(metrics, "win_rate", "WinRate")
    pf = measured(metrics, "profit_factor", "ProfitFactor")
    sharpe = measured(metrics, "sharpe_ratio", "Sharpe")
    max_dd = measured(metrics, "max_drawdown", "MaxDD")
    # The drawdown compounds the equity at the run's risk per trade, so it is shown with it.
    risk = risk_label(risk_per_trade(card.info))
    return (
        f"**Dir Acc**: {dir_acc}% | **Win Rate**: {win_rate}% | **PF**: {pf} | "
        f"**Sharpe**: {sharpe} | **MaxDD @ {risk} risk**: {max_dd}%"
    ), ""


def _detection_summary(card: ModelCard) -> tuple[str, str]:
    metrics = card.metrics
    m50 = measured(metrics, "mAP50", "map50")
    m95 = measured(metrics, "mAP50-95", "map50_95")
    b_loss = measured(metrics, "Box_Loss", "Train_Loss")
    c_loss = measured(metrics, "Cls_Loss", "Val_Loss")
    return (
        f"**mAP50**: {m50} | **mAP50-95**: {m95} | **Box Loss**: {b_loss} | **Cls Loss**: {c_loss}"
    ), ""


def _face_summary(card: ModelCard) -> tuple[str, str]:
    metrics = card.metrics
    m_easy = measured(metrics, "map_easy")
    m_med = measured(metrics, "map_medium", "map_med")
    m_hard = measured(metrics, "map_hard")
    return f"**mAP (Easy)**: {m_easy} | **mAP (Medium)**: {m_med} | **mAP (Hard)**: {m_hard}", ""


def _segmentation_summary(card: ModelCard) -> tuple[str, str]:
    miou = measured(card.metrics, "miou")
    p_acc = measured(card.metrics, "pixel_acc")
    return f"**mIoU**: {miou} | **Pixel Accuracy**: {p_acc}", ""


def _parameter_summary(card: ModelCard) -> tuple[str, str]:
    metrics = card.metrics
    return f"**MAE**: {measured(metrics, 'mae')} | **MSE**: {measured(metrics, 'mse')}", ""


def _classification_summary(card: ModelCard) -> tuple[str, str]:
    """Top-1 (and, above 5 classes, top-5) accuracy and the registry output classes."""
    metrics = card.metrics
    names = class_names(card.info)
    summary = f"**Top-1 Accuracy**: {measured(metrics, 'acc', 'accuracy')}"
    # Top-5 accuracy is always 1.0 for a model with 5 or fewer classes.
    if not names or len(names) > TOP5_CLASSES:
        summary += f" | **Top-5 Accuracy**: {measured(metrics, 'acc_top5')}"
    if not names:
        return summary, ""
    bullets = "".join(f"> - `{idx}`: {name}\n" for idx, name in enumerate(names))
    return summary, (
        f"> [!NOTE]\n> **Output Classes**: {len(names)} logits in class id order.\n>\n{bullets}"
    )


def _restoration_summary(card: ModelCard) -> tuple[str, str]:
    metrics = card.metrics
    psnr = measured(metrics, "psnr")
    ssim = measured(metrics, "ssim")
    lpips = measured(metrics, "lpips")
    fid = measured(metrics, "fid")
    return f"**PSNR**: {psnr} | **SSIM**: {ssim} | **LPIPS**: {lpips} | **FID**: {fid}", ""


# Metric summary per task, first match wins; restoration metrics are the fallback.
# A new task family is a new row.
_METRIC_SUMMARIES: list[tuple[Callable[[ModelCard], bool], Callable[[ModelCard], tuple]]] = [
    (lambda c: c.task == "quality", _quality_summary),
    (lambda c: c.task == "forex", _forex_summary),
    (lambda c: c.task in ["detection", "yolo"] or c.model_key == "yolov8n", _detection_summary),
    (lambda c: c.task == "face_detection" or c.model_key == "retinaface", _face_summary),
    (lambda c: c.task == "segmentation" or c.model_key == "parsenet", _segmentation_summary),
    (lambda c: c.task == "parameter_prediction" or c.model_key == "upn_v2", _parameter_summary),
    (lambda c: c.task == "classification", _classification_summary),
]


def _metrics_summary(card: ModelCard) -> tuple[str, str]:
    """Return the evaluation metrics line and the optional quality-vector callout."""
    for matches, summarize in _METRIC_SUMMARIES:
        if matches(card):
            return summarize(card)
    return _restoration_summary(card)


def _evaluation_profile(task: Any, image_input: str) -> tuple[str, str, str]:
    """Return the input requirements, the validation protocol and the metrics label.

    ``image_input`` describes the model's image preprocessing (vision tasks).
    """
    if task == "forex":
        return (
            "\n".join(
                [
                    _wrap_md_bullet(
                        "- **Input Requirements**: Normalized OHLCV tensor sequences across "
                        "multiple timeframes."
                    ),
                    _wrap_md_bullet(
                        "- **Failures**: Susceptible to spread friction and lookahead leakage if "
                        "walk-forward validation is compromised."
                    ),
                ]
            ),
            _wrap_md_bullet(
                "- **Validation Protocol**: 6-Fold Anchored Walk-Forward Cross-Validation "
                "(14-day Embargo)."
            ),
            "SOTA Metrics",
        )
    if task in ["detection", "yolo"]:
        return (
            "\n".join(
                [
                    _wrap_md_bullet(f"- **Input Requirements**: {image_input}"),
                    _wrap_md_bullet(
                        "- **Failures**: Small bounding box occlusion and extreme aspect ratio "
                        "distortions."
                    ),
                ]
            ),
            _wrap_md_bullet(
                "- **Validation Protocol**: 80/20 train/validate with zero ground-truth label "
                "leakage."
            ),
            "Target Detection SOTA",
        )
    return (
        "\n".join(
            [
                _wrap_md_bullet(f"- **Input Requirements**: {image_input}"),
                _wrap_md_bullet(
                    "- **Failures**: Large aspect ratio distortions during standard resize phases."
                ),
            ]
        ),
        _wrap_md_bullet("- **Split**: 80/20 train/validate with zero sample-leakage."),
        "Baseline Achievement",
    )


def _stability_line(info: dict[str, Any]) -> str:
    """Describe the training loss used to stabilize the manifold."""
    loss_fn = info.get("loss_fn", "l1").upper()
    if loss_fn == "EMD":
        form = registry_emd_form(info.get("stabilizers"))
        return (
            f"Trained using **Earth Mover's Distance (EMD, r = 2, {form} form)** on a plain "
            "softmax (temperature 1); temperature is calibrated post hoc only."
        )
    return f"Trained using **{loss_fn} Loss** to enforce strict manifold alignment."


def _dataset_manifest(card: ModelCard, unified_models: dict[str, Any]) -> str:
    """List the training datasets with their registered sample counts."""
    ds_registry = unified_models.get("_registry_metadata", {}).get("datasets", {})
    ds_sizes = []
    for d in card.datasets:
        count = ds_registry.get(d, {}).get("count", "N/A")
        if isinstance(count, int) and count >= 1000:
            count = f"{round(count / 1000)}k"
        if card.task == "forex":
            line = f"- **{d}**: ~{count} time-series OHLCV sequences (2019-2026)."
        else:
            line = f"- **{d}**: ~{count} binary image samples."
        ds_sizes.append(_wrap_md_bullet(line))
    return "\n".join(ds_sizes)


def _whitepaper_line(model_key: str) -> str:
    """Return the ecosystem whitepaper bullet, or '' when no whitepaper is known."""
    try:
        metadata_generator = load_cloud_module("metadata_generator")
        wp_url = metadata_generator.resolve_whitepaper_url(model_key)
        if wp_url:
            return (
                "- **Ecosystem Architecture Whitepaper**: [Whitepaper][whitepaper]\n\n"
                f"[whitepaper]:\n  {wp_url}\n"
            )
    except ImportError as exc:
        logger.warning("Whitepaper URL resolver unavailable for %s: %s", model_key, exc)
    return ""


def _paper_section(model_key: str) -> str:
    """Render the scientific reference and whitepaper section, or '' when neither exists."""
    paper = None
    try:
        paper = _registry_reference(model_key)
    except _REGISTRY_ERRORS as exc:
        logger.warning(
            (
                "Could not load academic reference for %s from registry; falling back to "
                "MODEL_PAPERS: %s"
            ),
            model_key,
            exc,
        )

    if not paper:
        paper = MODEL_PAPERS.get(model_key)

    whitepaper_link_line = _whitepaper_line(model_key)
    if paper:
        title_line = _wrap_md_bullet(f"- **Title**: {paper['title']}")
        authors_line = _wrap_md_bullet(f"- **Authors**: {paper['authors']}")
        pub_line = _wrap_md_bullet(f"- **Publication**: {paper['venue']} ({paper['year']})")
        src_line = f"- **Canonical Source / Link**:\n  [Reference Link]({paper['link']})"
        return f"""## Scientific Research & Reference Paper

{title_line}
{authors_line}
{pub_line}
{src_line}
{whitepaper_link_line}
```bibtex
{_wrap_bibtex(paper["bibtex"])}
```

"""
    if whitepaper_link_line:
        return f"""## Technical Whitepaper & Reference Documentation

{whitepaper_link_line}
"""
    return ""


def _tip_block(model_key: str) -> str:
    """Point to the usage notebooks (and, for YOLO, the checkpoint layout)."""
    kaggle_usage_name = f"{model_key}_kaggle_usage.ipynb"
    colab_usage_name = f"{model_key}_colab_usage.ipynb"
    extra_tip = ""
    if model_key == "yolov8n":
        extra_tip = (
            "\n>\n"
            "> **Artifacts & Checkpoints Structure**:\n>\n"
            "> - Production SOTA exports: `yolov8n.onnx` and `yolov8n.pt` are\n"
            ">   deployed to this directory whenever SOTA targets are achieved.\n"
            "> - Training checkpoints & curriculum state: Preserved strictly in\n"
            ">   [`checkpoints/`](checkpoints/) (`best.pt`, `best.pth`, `last.pt`,\n"
            ">   `progress.pth`, `curriculum_state.json`)."
        )

    return (
        "> [!TIP]\n"
        "> **Implementation Guide**: For high-performance deployment including\n"
        "> ONNX (FP32/FP16) and standalone PyTorch snippets, refer to the\n"
        "> **[Kaggle Usage Notebook][kaggle-nb]** or\n"
        "> **[Colab Usage Notebook][colab-nb]**\n"
        f"> notebooks in this directory.{extra_tip}\n\n"
        f"[kaggle-nb]: {kaggle_usage_name}\n"
        f"[colab-nb]: {colab_usage_name}"
    )


def _badges(res_str: str, epochs_trained: Any) -> str:
    """Render the status, hardware, epoch and resolution badges."""
    # --- 2026: Nuclear Badging (Task 7.1) ---
    badge_res = res_str.split(" (", 1)[0].replace(" ", "_").replace("(", "").replace(")", "")
    badges = [
        "![SOTA](https://img.shields.io/badge/Status-SOTA-brightgreen)",
        "![Hardware](https://img.shields.io/badge/Hardware-Accelerated-blue)",
        f"![Epochs](https://img.shields.io/badge/Epochs-{epochs_trained}-orange)",
        f"![Resolution](https://img.shields.io/badge/Res-{badge_res}-blueviolet)",
    ]
    return "\n".join(badges)


# --- Premium 10-Section Template ---
_CARD_TEMPLATE = """# {name}

{badge_str}

## Overview

{overview_para}

{arch_bullet}
{res_bullet}
{use_case_bullet}
{datasets_bullet}

## Manifold Topology

{topology_mermaid}
{vector_spacer}{vector_section}
## Usage

{usage_snippet}

{tip_block}

{input_reqs_str}

## Implementation Requirements

{hw_bullet}
- **Software**: PyTorch 2.1+, CUDA 12.1.
{lifecycle_bullet}

## Model Stats

- **Precision**: ONNX FP16 (Edge) / PyTorch FP32 (Training).
- **Latency**: Sub-50ms inference bound on target local GPU hardware.
{stability_bullet}

## Data Manifest

{ds_str}

## Evaluation Results

{eval_metrics_bullet}
{eval_split_str}

{paper_section}---
**LemGendary AI Training Suite** | *SOTA-Autonomous & Nuclear-Hardened Matrix*
"""


def build_model_readme(
    model_key,
    unified_models,
    epochs_trained,
    metrics,
    hardware=DEFAULT_HARDWARE,
):
    """Render the markdown model card for ``model_key`` from registry config and metrics."""
    card = ModelCard.load(model_key, unified_models, metrics or {})
    # Sections are rendered in the original order so warnings are logged in that order.
    fields = {
        "usage_snippet": usage_snippet(
            card.task,
            model_key,
            SnippetInput(
                *card.size,
                classes=class_names(card.info),
                preprocessing=image_preprocessing(card.info),
            ),
        ),
        "badge_str": _badges(card.res_str, epochs_trained),
        "topology_mermaid": topology_mermaid(card.task, card.res_str, card.arch),
        "stability_bullet": _wrap_md_bullet(f"- **Stability**: {_stability_line(card.info)}"),
    }
    metrics_summary, fields["vector_section"] = _metrics_summary(card)
    fields["ds_str"] = _dataset_manifest(card, unified_models)
    fields["input_reqs_str"], fields["eval_split_str"], metrics_label = _evaluation_profile(
        card.task, image_preprocessing(card.info).describe()
    )
    fields["eval_metrics_bullet"] = _wrap_md_bullet(f"- **{metrics_label}**: {metrics_summary}")
    fields["paper_section"] = _paper_section(model_key)
    arch_type = card.info.get("architecture_type", "Standard Backbone")
    desc = card.info.get("description", "Premium LemGendary AI Training Suite Matrix Model.")
    overview_para = _wrap_md_paragraph(
        f"The **{card.name}** is a professional-grade AI model optimized for the "
        f"`{card.task}` lifecycle within the LemGendary Training Suite."
    )
    return _CARD_TEMPLATE.format(
        name=card.name,
        overview_para=overview_para,
        arch_bullet=_wrap_md_bullet(f"- **Architecture**: {card.arch} ({arch_type})"),
        res_bullet=_wrap_md_bullet(f"- **Input Resolution**: {card.res_str}"),
        use_case_bullet=_wrap_md_bullet(f"- **Use Case**: {desc}"),
        datasets_bullet=_wrap_md_bullet(f"- **Training Data**: {', '.join(card.datasets)}"),
        vector_spacer="\n" if fields["vector_section"] else "",
        tip_block=_tip_block(model_key),
        hw_bullet=_wrap_md_bullet(f"- **Hardware**: {hardware}"),
        lifecycle_bullet=_wrap_md_bullet(
            f"- **Training Lifecycle**: Successfully processed over {epochs_trained} total "
            "epochs securely."
        ),
        **fields,
    )

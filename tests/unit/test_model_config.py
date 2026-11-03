"""Unit tests for model constructor configs built from registry ``model_kwargs``."""

from __future__ import annotations

import unittest

from models.core_restoration import NAFNet, NAFNetConfig
from models.forex_predictor import ForexArch, ForexPredictor
from models.generic_causal_llm import GenericCausalLLM, LoraSettings
from models.model_config import resolve_config


def _shapes(model) -> dict[str, tuple[int, ...]]:
    """Parameter name -> shape, to compare two builds of a model."""
    return {name: tuple(p.shape) for name, p in model.state_dict().items()}


class TestResolveConfig(unittest.TestCase):
    """``resolve_config`` maps flat keywords onto a config dataclass."""

    def test_keywords_fill_fields_and_none_keeps_default(self) -> None:
        """Field keywords are used, ``None`` and unknown keywords are not."""
        cfg = resolve_config(NAFNetConfig, None, {"width": 8, "out_ch": None, "other": 1})
        self.assertEqual(cfg, NAFNetConfig(width=8))

    def test_config_wins_and_conflicts_raise(self) -> None:
        """An explicit config is returned; mixing it with field keywords raises."""
        cfg = NAFNetConfig(width=8)
        self.assertIs(resolve_config(NAFNetConfig, cfg, {"other": 1}), cfg)
        with self.assertRaises(TypeError):
            resolve_config(NAFNetConfig, cfg, {"width": 16})


class TestModelConfigs(unittest.TestCase):
    """Models build the same layers from a config or from flat ``model_kwargs``."""

    def test_nafnet_flat_kwargs_match_config(self) -> None:
        """``NAFNet(**model_kwargs)`` equals ``NAFNet(config=...)``."""
        kwargs = {"width": 8, "enc_blk_nums": [1, 1], "dec_blk_nums": [1, 1]}
        flat = NAFNet(**kwargs)
        typed = NAFNet(config=NAFNetConfig(width=8, enc_blk_nums=(1, 1), dec_blk_nums=(1, 1)))
        self.assertEqual(_shapes(flat), _shapes(typed))
        self.assertEqual(len(flat.encoders), 2)

    def test_forex_flat_kwargs_match_arch(self) -> None:
        """``ForexPredictor(**model_kwargs)`` equals ``ForexPredictor(arch=...)``."""
        flat = ForexPredictor([1, 5], d_model=32, n_heads=2, n_layers=1, head_hidden=16)
        typed = ForexPredictor(
            [1, 5], arch=ForexArch(d_model=32, n_heads=2, n_layers=1, head_hidden=16)
        )
        self.assertEqual(flat.arch, typed.arch)
        self.assertEqual(_shapes(flat), _shapes(typed))

    def test_causal_llm_lora_settings(self) -> None:
        """LoRA keywords and ``LoraSettings`` set the same adapter switch."""
        flat = GenericCausalLLM("mock", use_lora=False)
        typed = GenericCausalLLM("mock", lora=LoraSettings(use_lora=False))
        self.assertFalse(flat.use_lora)
        self.assertFalse(typed.use_lora)


if __name__ == "__main__":
    unittest.main()

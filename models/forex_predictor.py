"""ForexPredictor: multi-timeframe causal-conv/attention model for FX direction and magnitude."""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from data.forex_spec import EXTENDED_PAIRS, NUM_PAIRS, TIMEFRAME_LOOKBACK

logger = logging.getLogger("lemtrain.models.forex_predictor")

# [LemGendary Forex Suite v1.1 - SYNC_ID: FOREX_02]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# OHLCV + 5 technical indicators (RSI, MACD, Signal, ATR, BBW)
# + 2 session encoding (sin/cos hour) + 2 volatility regime (ATR percentile, bar range ratio)
# = 14 features per bar
FOREX_FEATURES_PER_BAR = 14

# Direction classes: 0=Down, 1=Sideways, 2=Up
DIRECTION_CLASSES = 3

# 2026 v1.1: Single source of truth for magnitude ceiling.
# Must equal ForexDualLoss.max_pips (100.0) so no gradient signal is silently discarded
# during training. The pipeline normalizes pip targets into the ~[10, 100] NPU range,
# so 100 pips is the correct physical ceiling for both the head and the loss.
MAX_PIPS_CEILING = 100.0


def _select_valid_n_heads(d_model: int, n_tf: int, preferred: int) -> int:
    """
    2026 v1.1 Fix: nn.MultiheadAttention requires embed_dim % num_heads == 0.
    Picks the largest preferred divisor of d_model that is <= n_tf.
    Falls back to 1 (always valid) if nothing else fits.
    """
    candidates = [preferred, 8, 6, 4, 3, 2, 1]
    for h in candidates:
        if h <= n_tf and d_model % h == 0:
            return h
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────


class CausalConv1DBlock(nn.Module):
    """
    Causal (no future lookahead) dilated Conv1D block with stochastic depth.
    Padding is left-only to ensure strict temporal causality for live trading.
    Stochastic depth randomly skips the block during training (drop-path regularization)
    to prevent the model from memorizing specific temporal patterns.

    The convolution kernel is :attr:`KERNEL_SIZE` (3) wide.

    Args:
        in_channels:    Input feature depth.
        out_channels:   Output feature depth.
        dilation:       Dilation factor for exponential receptive field growth.
        dropout:        Dropout probability.
        drop_path_rate: Probability of dropping the entire block (stochastic depth).
    """

    KERNEL_SIZE = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        *,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        kernel_size = self.KERNEL_SIZE
        # Left-only padding to prevent future data leakage
        self.causal_pad = (kernel_size - 1) * dilation
        self.drop_path_rate = drop_path_rate

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Projection shortcut when channel dims differ
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the causal conv block with optional stochastic depth; ``x`` is [B, C, T]."""
        # x: [B, C, T]
        residual = self.shortcut(x)
        # Stochastic depth: randomly drop entire block during training
        if (
            self.training
            and self.drop_path_rate > 0.0
            and torch.rand(1).item() < self.drop_path_rate
        ):
            return residual
        x = F.pad(x, (self.causal_pad, 0))  # Causal left-pad only
        x = self.conv(x)  # [B, out_channels, T]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm over channels
        x = self.act(x)
        x = self.drop(x)
        # Scale by survival probability at inference to match expected activation
        if not self.training and self.drop_path_rate > 0.0:
            x = x * (1.0 - self.drop_path_rate)
        return x + residual


class TimeframeEncoder(nn.Module):
    """
    Per-timeframe TCN encoder: stacks CausalConv1D blocks with exponential dilation.
    Produces a fixed-length feature vector for each timeframe via temporal pooling.
    Supports optional pair embedding injection for pair-specific local pattern learning.
    The encoder works on any lookback length (``TIMEFRAME_LOOKBACK`` sets it per timeframe).

    Args:
        in_features:  Number of input features per bar (default: FOREX_FEATURES_PER_BAR).
        d_model:      Output embedding dimension.
        n_layers:     Number of CausalConv1D layers.
        dropout:      Dropout probability.
        pair_d_model: Dimension of pair embedding to inject as input bias (0 = disabled).
    """

    def __init__(
        self,
        in_features: int = FOREX_FEATURES_PER_BAR,
        d_model: int = 192,
        n_layers: int = 4,
        dropout: float = 0.1,
        pair_d_model: int = 0,
    ):
        super().__init__()
        self.pair_d_model = pair_d_model

        # Pair embedding projection: maps [B, d_pair] -> [B, in_features] for additive input bias
        if pair_d_model > 0:
            self.pair_proj = nn.Linear(pair_d_model, in_features)
        else:
            self.pair_proj = None

        layers = []
        in_ch = in_features
        for i in range(n_layers):
            dilation = 2**i
            layers.append(CausalConv1DBlock(in_ch, d_model, dilation, dropout=dropout))
            in_ch = d_model
        self.tcn = nn.Sequential(*layers)

        # Multi-scale pooling: concat mean + last-step -> 2*d_model
        self.project = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pair_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:        [B, seq_len, in_features]
            pair_emb: [B, pair_d_model] optional pair embedding for input conditioning
        Returns:
            feat: [B, d_model]
        """
        if pair_emb is not None and self.pair_proj is not None:
            # Inject pair embedding as additive bias across all time steps
            pair_bias = self.pair_proj(pair_emb).unsqueeze(1)  # [B, 1, in_features]
            x = x + pair_bias  # [B, seq_len, in_features]
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.tcn(x)  # [B, d_model, T]

        mean_pool = x.mean(dim=2)  # [B, d_model]  -- macro context
        last_step = x[:, :, -1]  # [B, d_model]  -- most recent state
        pooled = torch.cat([mean_pool, last_step], dim=1)  # [B, 2*d_model]

        feat = self.project(pooled)
        feat = self.norm(feat)
        return feat


class CrossTimeframeAttention(nn.Module):
    """
    Multi-head self-attention over the set of timeframe embeddings.
    Learns which timeframes carry the most predictive signal for the current bar.
    Includes Timeframe Dropout during training to prevent single-timeframe co-adaptation.

    Args:
        d_model:    Embedding dimension per timeframe.
        n_heads:    Number of attention heads (must divide d_model evenly).
        dropout:    Dropout probability.
        tf_dropout: Timeframe dropout probability during training.
    """

    def __init__(
        self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1, tf_dropout: float = 0.15
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.tf_dropout = tf_dropout

    def forward(self, tf_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tf_feats: [B, num_timeframes, d_model]
        Returns:
            out: [B, num_timeframes * d_model]  (flattened for head input)
        """
        if self.training and self.tf_dropout > 0.0 and tf_feats.shape[1] > 1:
            mask = (
                torch.rand(tf_feats.shape[0], tf_feats.shape[1], 1, device=tf_feats.device)
                >= self.tf_dropout
            ).float()
            all_masked = mask.sum(dim=1, keepdim=True) == 0
            if all_masked.any():
                rand_idx = torch.randint(
                    0, tf_feats.shape[1], (tf_feats.shape[0],), device=tf_feats.device
                )
                for b in range(tf_feats.shape[0]):
                    if all_masked[b, 0, 0]:
                        mask[b, rand_idx[b], 0] = 1.0
            tf_feats = tf_feats * mask

        residual = tf_feats
        attn_out, _ = self.attn(tf_feats, tf_feats, tf_feats)
        out = self.norm(residual + self.drop(attn_out))  # [B, n_tf, d_model]
        return out.flatten(1)  # [B, n_tf * d_model]


class DirectionHead(nn.Module):
    """
    3-class direction head: Down (0), Sideways (1), Up (2).
    Returns raw logits (softmax applied externally for loss / argmax for inference).

    Args:
        in_features: Fused manifold width.
        hidden_dim:  Hidden layer dimension.
        dropout:     Dropout probability.
    """

    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, DIRECTION_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return direction logits of shape [B, 3]."""
        return self.head(x)  # [B, 3]


class MagnitudeHead(nn.Module):
    """
    Dual regression head predicting expected TP and SL distances in pips.
    Outputs are constrained positive via Softplus and clamped to MAX_PIPS_CEILING
    (100 pips, matching ForexDualLoss) to keep gradient magnitudes bounded.

    Args:
        in_features: Fused manifold width.
        hidden_dim:  Hidden layer dimension.
        dropout:     Dropout probability.
    """

    def __init__(self, in_features: int, hidden_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tp_head = nn.Linear(hidden_dim, 1)  # Take-Profit pips
        self.sl_head = nn.Linear(hidden_dim, 1)  # Stop-Loss pips
        self.act = nn.Softplus()  # Ensures positive pip outputs
        # 2026 v1.1: Sourced from the module-level constant for loss-head parity
        self.max_pips = MAX_PIPS_CEILING

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            mag: [B, 2]  -- (tp_pips, sl_pips), positive, clamped to max_pips
        """
        h = self.trunk(x)
        tp = torch.clamp(self.act(self.tp_head(h)), max=self.max_pips)
        sl = torch.clamp(self.act(self.sl_head(h)), max=self.max_pips)
        return torch.cat([tp, sl], dim=1)  # [B, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ForexArch:
    """Architecture hyper-parameters of a ``ForexPredictor`` (its checkpoint ``model_kwargs``)."""

    d_model: int
    n_layers: int
    dropout: float
    tf_dropout: float
    in_features: int
    head_hidden: int


def _forex_layers(
    arch: ForexArch, active_timeframes: list, num_pairs: int, n_heads: int
) -> dict[str, nn.Module]:
    """ForexPredictor children in registration order (the names are the ``state_dict`` keys)."""
    d_model = arch.d_model
    n_tf = len(active_timeframes)
    # Fused manifold dimension: n_tf * d_model (from cross_attn) + d_model (pair embed)
    fused_dim = n_tf * d_model + d_model
    return {
        # Pair embedding: early projection for TCN input conditioning
        "pair_embed": nn.Embedding(num_pairs, d_model),
        "pair_project": nn.Linear(d_model, d_model),
        # One TCN encoder per active timeframe; each receives the pair embedding as an
        # additive input bias.
        "encoders": nn.ModuleDict(
            {
                str(tf): TimeframeEncoder(
                    in_features=arch.in_features,
                    d_model=d_model,
                    n_layers=arch.n_layers,
                    dropout=arch.dropout,
                    pair_d_model=d_model,
                )
                for tf in active_timeframes
            }
        ),
        # Cross-timeframe attention fuses multiple timeframe embeddings.
        # 2026 v1.1 Fix: guarantee d_model % n_heads == 0 for nn.MultiheadAttention
        "cross_attn": CrossTimeframeAttention(
            d_model=d_model,
            n_heads=_select_valid_n_heads(d_model, n_tf, preferred=n_heads),
            dropout=arch.dropout,
            tf_dropout=arch.tf_dropout,
        ),
        # Manifold projection before heads
        "fused_project": nn.Sequential(
            nn.Linear(fused_dim, arch.head_hidden),
            nn.GELU(),
            nn.Dropout(arch.dropout),
            nn.LayerNorm(arch.head_hidden),
        ),
        "direction_head": DirectionHead(arch.head_hidden, arch.head_hidden, arch.dropout),
        "magnitude_head": MagnitudeHead(arch.head_hidden, arch.head_hidden, arch.dropout),
    }


class ForexPredictor(nn.Module):
    """
    LemGendary Forex Predictor -- Multi-Scale CNN-Transformer Hybrid.

    Architecture:
        1. Per-timeframe TCN encoders (CausalConv1D stacks, 4 layers each, stochastic depth)
        2. Pair embedding injected at TCN encoder input for pair-specific local pattern learning
        3. Cross-timeframe attention (learns which TF is most relevant per bar)
        4. Currency pair embedding fused into the final manifold
        5. Dual output heads:
           - DirectionHead:  3-class logits [Down, Sideways, Up]
           - MagnitudeHead:  [TP pips, SL pips] clamped to MAX_PIPS_CEILING (100)

    Design constraints:
        - Fully stateless (no hidden LSTM state) -> safe for ONNX + MT5 EA
        - Causal convolutions only (no future lookahead)
        - ONNX-compatible (no custom ops)
        - Active timeframes are controlled by the Governor's seq_len rung

    v1.1 Fixes:
        - pair_idx=None no longer crashes (safe zero-index fallback via _safe_pair_idx).
        - cross_attn uses _select_valid_n_heads() so d_model % n_heads == 0 is guaranteed
          even when the curriculum lands on 5 timeframes.
        - expand_timeframe() warm-starts cross_attn via load_state_dict and partially
          copies the old fused_project input weights into the wider replacement.
        - expand_timeframe() preserves pair_d_model conditioning on the new branch.
        - MagnitudeHead.max_pips is synchronized with ForexDualLoss.max_pips (100).
        - pair_idx is clamped to [0, num_pairs-1] to prevent nn.Embedding IndexError.

    Arguments after ``active_timeframes`` are keyword-only; the model factory and the
    checkpoint ``model_kwargs`` pass them by name.

    Args:
        active_timeframes: List of active timeframe rungs (minutes).
                           Curriculum: start with [1], expand to all 6.
        d_model:           Embedding dimension per timeframe branch.
        n_heads:           Preferred number of cross-timeframe attention heads.
                           Clamped to the largest divisor of d_model that is <= n_tf.
        n_layers:          Number of CausalConv1D layers per encoder.
        head_hidden:       Hidden dim for both output heads.
        dropout:           Global dropout rate.
        tf_dropout:        Timeframe dropout rate for cross-attention.
        in_features:       Features per bar (default: FOREX_FEATURES_PER_BAR = 14).
        pairs:             Optional subset of pairs for pair-embedding sizing.
    """

    # Children registered by name in ``__init__`` (see ``_forex_layers``); the names are
    # the checkpoint ``state_dict`` prefixes.
    pair_embed: nn.Embedding
    pair_project: nn.Linear
    encoders: nn.ModuleDict
    cross_attn: CrossTimeframeAttention
    fused_project: nn.Sequential
    direction_head: DirectionHead
    magnitude_head: MagnitudeHead

    # Keyword arguments are the registry ``model_kwargs`` stored with each checkpoint.
    def __init__(  # pylint: disable=too-many-arguments
        self,
        active_timeframes: list | None = None,
        *,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        head_hidden: int = 384,
        dropout: float = 0.1,
        tf_dropout: float = 0.15,
        in_features: int = FOREX_FEATURES_PER_BAR,
        pairs: list | None = None,
        **kwargs,
    ):
        super().__init__()

        if active_timeframes is None:
            active_timeframes = [1, 5, 15, 60, 240, 1440]

        self.active_timeframes = list(active_timeframes)
        self.arch = ForexArch(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            tf_dropout=tf_dropout,
            in_features=in_features,
            head_hidden=head_hidden,
        )

        self.pairs = list(pairs) if pairs is not None else list(EXTENDED_PAIRS)
        self.num_pairs = max(len(self.pairs), NUM_PAIRS)

        for name, layer in _forex_layers(
            self.arch, self.active_timeframes, self.num_pairs, n_heads
        ).items():
            self.add_module(name, layer)

        self._init_weights()

    @property
    def in_features(self) -> int:
        """Features per bar the timeframe encoders expect (read by the MT5 exporter)."""
        return self.arch.in_features

    def _init_weights(self):
        """Kaiming init for linear/conv layers, Xavier for embeddings."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.pair_embed.weight)

    def _safe_pair_idx(
        self,
        pair_idx: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        2026 v1.1: Accept None, scalars, or tensors. Always returns a [B] long tensor
        of valid embedding indices in [0, num_pairs-1].
        """
        if pair_idx is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if not isinstance(pair_idx, torch.Tensor):
            pair_idx = torch.as_tensor(pair_idx, dtype=torch.long, device=device)
        pair_idx = pair_idx.to(device=device, dtype=torch.long)
        if pair_idx.dim() == 0:
            pair_idx = pair_idx.unsqueeze(0).expand(batch_size)
        # Clamp to safe range for nn.Embedding
        pair_idx = pair_idx.clamp(min=0, max=self.num_pairs - 1)
        return pair_idx

    def _encode_timeframes(self, tf_inputs: dict, pair_emb: torch.Tensor) -> list[torch.Tensor]:
        """Encode each active timeframe with its pair-conditioned TCN, in rung order."""
        tf_feats = []
        for tf in self.active_timeframes:
            encoder_key = str(tf)
            val = tf_inputs.get(tf, tf_inputs.get(encoder_key))

            if val is None or encoder_key not in self.encoders:
                # 2026 v1.1 Fix: Fail loudly instead of silently duplicating features.
                # A silent duplication here would corrupt the training signal by
                # presenting two slots with the same underlying tensor.
                raise KeyError(
                    f"[ForexPredictor] Missing timeframe {tf} in batch or encoder. "
                    f"Provided input keys: {list(tf_inputs.keys())}, "
                    f"Encoder keys: {list(self.encoders.keys())}"
                )

            tf_feats.append(self.encoders[encoder_key](val, pair_emb))
        return tf_feats

    def forward(
        self,
        tf_inputs: dict,
        pair_idx: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            tf_inputs: Dict mapping timeframe (int) -> tensor [B, seq_len, features]
                       Only active timeframes need to be present.
            pair_idx:  Optional Long tensor [B] -- currency pair index (see PAIR_INDEX).
                       Falls back to zeros if None.

        Returns:
            Dict with:
                'direction_logits': [B, 3]  -- raw logits for CE loss / argmax
                'magnitude':        [B, 2]  -- (tp_pips, sl_pips)
        """
        # Infer batch size from any provided timeframe tensor
        first_val = next(iter(tf_inputs.values()))
        batch_size = first_val.shape[0]
        device = first_val.device

        # 2026 v1.1 Fix: None-safe pair_idx handling
        pair_idx = self._safe_pair_idx(pair_idx, batch_size, device)

        # Pair embedding: used both for TCN input conditioning and final manifold fusion
        pair_emb_raw = self.pair_embed(pair_idx)  # [B, d_model]
        pair_emb = self.pair_project(pair_emb_raw)  # [B, d_model]

        # Stack -> [B, n_tf, d_model] -> cross-timeframe attention -> [B, n_tf * d_model]
        tf_stack = torch.stack(self._encode_timeframes(tf_inputs, pair_emb), dim=1)
        fused_tf = self.cross_attn(tf_stack)  # [B, n_tf * d_model]

        # Final manifold fusion with pair embedding
        fused = torch.cat([fused_tf, pair_emb], dim=1)  # [B, n_tf*d_model + d_model]
        manifold = self.fused_project(fused)  # [B, head_hidden]

        direction = self.direction_head(manifold)  # [B, 3]
        magnitude = self.magnitude_head(manifold)  # [B, 2]

        return {
            "direction": direction,
            "direction_logits": direction,
            "magnitude": magnitude,
        }

    def expand_timeframe(self, new_tf: int):
        """
        Governor hook: add a new timeframe branch without rebuilding the entire model.
        Called by the Governor when advancing to the next curriculum rung.

        2026 v1.1: Warm-starts cross_attn (via load_state_dict) and copies the
        old fused_project input weights into the new (wider) fused_project so
        prior learning is preserved across the curriculum jump.

        Args:
            new_tf: Timeframe in minutes to add (must be in TIMEFRAME_LOOKBACK).
        """
        if new_tf in self.active_timeframes:
            return
        if new_tf not in TIMEFRAME_LOOKBACK:
            raise ValueError(
                f"[ForexPredictor] Unknown timeframe: {new_tf}. "
                f"Valid: {list(TIMEFRAME_LOOKBACK.keys())}"
            )

        key = str(new_tf)

        # 2026 v1.1 Fix: inherit pair_d_model from an existing encoder so the
        # new branch stays pair-conditioned like its siblings.
        ref_encoder = next(iter(self.encoders.values()))
        ref_pair_d_model = getattr(ref_encoder, "pair_d_model", self.arch.d_model)

        self.encoders[key] = TimeframeEncoder(
            in_features=self.arch.in_features,
            d_model=self.arch.d_model,
            n_layers=self.arch.n_layers,
            dropout=self.arch.dropout,
            pair_d_model=ref_pair_d_model,
        )
        self.active_timeframes.append(new_tf)

        n_tf = len(self.active_timeframes)
        self._rebuild_cross_attn(n_tf)
        self._rebuild_fused_project(n_tf, new_tf)

    def _rebuild_cross_attn(self, n_tf: int) -> None:
        """Replace ``cross_attn`` for ``n_tf`` timeframes, warm-started from the old one."""
        old_cross_attn = self.cross_attn
        old_cross_state = {k: v.clone() for k, v in old_cross_attn.state_dict().items()}

        # 2026 v1.1 Fix: use the divisibility helper (was relying on nn.MultiheadAttention
        # to raise an obscure AssertionError on non-divisible configs).
        valid_n_heads = _select_valid_n_heads(
            self.arch.d_model, n_tf, preferred=old_cross_attn.attn.num_heads
        )

        self.add_module(
            "cross_attn",
            CrossTimeframeAttention(
                d_model=self.arch.d_model,
                n_heads=valid_n_heads,
                tf_dropout=self.arch.tf_dropout,
            ),
        )

        # 2026 v1.1 Warm-Start: MultiheadAttention's in_proj / out_proj / LayerNorm
        # weights are independent of n_tf, so they load cleanly. strict=False lets
        # any structurally-incompatible keys fall back to fresh init.
        try:
            self.cross_attn.load_state_dict(old_cross_state, strict=False)
            print(
                f" [ForexPredictor] cross_attn warm-started (heads={valid_n_heads}, n_tf={n_tf})."
            )
        except RuntimeError as e:
            # load_state_dict raises RuntimeError on tensor shape mismatches.
            logger.warning(
                "cross_attn warm-start skipped (heads=%d, n_tf=%d): %s", valid_n_heads, n_tf, e
            )

    def _rebuild_fused_project(self, n_tf: int, new_tf: int) -> None:
        """Widen ``fused_project`` for ``n_tf`` timeframes, keeping the old input weights.

        Leaves ``fused_project`` untouched when it is not the Linear/GELU/Dropout/LayerNorm
        stack that ``__init__`` builds.
        """
        old_proj = self.fused_project
        if not isinstance(old_proj, nn.Sequential) or len(old_proj) < 4:
            return
        m0 = old_proj[0]
        m2 = old_proj[2]
        m3 = old_proj[3]
        if not (
            isinstance(m0, nn.Linear)
            and isinstance(m2, nn.Dropout)
            and isinstance(m3, nn.LayerNorm)
        ):
            return

        fused_dim = n_tf * self.arch.d_model + self.arch.d_model
        new_linear = nn.Linear(fused_dim, m0.out_features)
        new_ln = nn.LayerNorm(m0.out_features)
        new_proj = nn.Sequential(new_linear, nn.GELU(), nn.Dropout(float(m2.p)), new_ln)
        _copy_projection_weights(m0, m3, new_linear, new_ln)
        self.add_module("fused_project", new_proj)

        print(f" [ForexPredictor] Expanded to {n_tf} timeframes. Added: {new_tf}min rung.")
        print(
            " [ForexPredictor] fused_project warm-started: "
            f"old_in_dim={m0.in_features} -> new_in_dim={fused_dim}."
        )


def _copy_projection_weights(
    old_linear: nn.Linear, old_norm: nn.LayerNorm, new_first_linear: nn.Linear, new_ln: nn.LayerNorm
) -> None:
    """Copy the old input-block weights and LayerNorm into a wider ``fused_project``.

    2026 v1.1: the fused_tf layout is [n_tf * d_model, pair_emb(d_model)], so the old
    timeframe block + the pair embedding block occupy the FIRST ``in_features`` rows of
    the old linear. The new d_model rows at the tail belong to the new timeframe and
    keep their Kaiming initialization.
    """
    with torch.no_grad():
        new_first_linear.weight[: old_linear.in_features] = old_linear.weight
        if new_first_linear.bias is not None and old_linear.bias is not None:
            new_first_linear.bias.copy_(old_linear.bias)
        # LayerNorm at the tail is shape-invariant (hidden-dim); copy directly
        if (
            old_norm.weight is not None
            and new_ln.weight is not None
            and old_norm.weight.shape == new_ln.weight.shape
        ):
            new_ln.weight.copy_(old_norm.weight)
            if new_ln.bias is not None and old_norm.bias is not None:
                new_ln.bias.copy_(old_norm.bias)

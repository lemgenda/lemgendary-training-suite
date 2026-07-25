import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# [LemGendary Forex Suite v1.0 - SYNC_ID: FOREX_01]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# OHLCV + 5 technical indicators (RSI, MACD, Signal, ATR, BBW) = 10 features per bar
FOREX_FEATURES_PER_BAR = 10

# Direction classes: 0=Down, 1=Sideways, 2=Up
DIRECTION_CLASSES = 3

# Timeframe rungs in minutes — mirrors Governor's res_ladder
TIMEFRAME_RUNGS = [1, 5, 15, 60, 240, 1440]

# Canonical lookback windows per timeframe (covers ~1 trading week each)
TIMEFRAME_LOOKBACK = {
    1:    512,   # M1  → ~8.5 hours
    5:    288,   # M5  → ~1 day
    15:   192,   # M15 → ~2 days
    60:   168,   # H1  → ~1 week
    240:   90,   # H4  → ~2.5 weeks
    1440:  252,  # D1  → ~1 year
}

# Major currency pair index (0-based, used for pair embedding)
PAIR_INDEX = {
    "EURUSD": 0, "GBPUSD": 1, "USDJPY": 2, "USDCHF": 3,
    "AUDUSD": 4, "USDCAD": 5, "NZDUSD": 6, "XAUUSD": 7,
}
NUM_PAIRS = len(PAIR_INDEX)


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1DBlock(nn.Module):
    """
    Causal (no future lookahead) dilated Conv1D block.
    Padding is left-only to ensure strict temporal causality for live trading.

    Args:
        in_channels:  Input feature depth.
        out_channels: Output feature depth.
        kernel_size:  Convolution kernel width.
        dilation:     Dilation factor for exponential receptive field growth.
        dropout:      Dropout probability.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Left-only padding to prevent future data leakage
        self.causal_pad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Projection shortcut when channel dims differ
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = self.shortcut(x)
        x = F.pad(x, (self.causal_pad, 0))       # Causal left-pad only
        x = self.conv(x)                           # [B, out_channels, T]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm over channels
        x = self.act(x)
        x = self.drop(x)
        return x + residual


class TimeframeEncoder(nn.Module):
    """
    Per-timeframe TCN encoder: stacks CausalConv1D blocks with exponential dilation.
    Produces a fixed-length feature vector for each timeframe via temporal pooling.

    Args:
        seq_len:     Lookback window length (number of bars).
        in_features: Number of input features per bar (default: FOREX_FEATURES_PER_BAR).
        d_model:     Output embedding dimension.
        n_layers:    Number of CausalConv1D layers.
        dropout:     Dropout probability.
    """
    def __init__(
        self,
        seq_len: int,
        in_features: int = FOREX_FEATURES_PER_BAR,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len

        layers = []
        in_ch = in_features
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(CausalConv1DBlock(in_ch, d_model, kernel_size=3, dilation=dilation, dropout=dropout))
            in_ch = d_model
        self.tcn = nn.Sequential(*layers)

        # Multi-scale pooling: concat mean + last-step → 2*d_model
        self.project = nn.Linear(2 * d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, in_features]
        Returns:
            feat: [B, d_model]
        """
        x = x.transpose(1, 2)          # [B, F, T]
        x = self.tcn(x)                 # [B, d_model, T]

        mean_pool = x.mean(dim=2)       # [B, d_model]  — macro context
        last_step = x[:, :, -1]         # [B, d_model]  — most recent state
        pooled    = torch.cat([mean_pool, last_step], dim=1)  # [B, 2*d_model]

        feat = self.project(pooled)
        feat = self.norm(feat)
        return feat


class CrossTimeframeAttention(nn.Module):
    """
    Multi-head self-attention over the set of timeframe embeddings.
    Learns which timeframes carry the most predictive signal for the current bar.

    Args:
        d_model:  Embedding dimension per timeframe.
        n_heads:  Number of attention heads.
        dropout:  Dropout probability.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tf_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tf_feats: [B, num_timeframes, d_model]
        Returns:
            out: [B, num_timeframes * d_model]  (flattened for head input)
        """
        residual = tf_feats
        attn_out, _ = self.attn(tf_feats, tf_feats, tf_feats)
        out = self.norm(residual + self.drop(attn_out))  # [B, n_tf, d_model]
        return out.flatten(1)                             # [B, n_tf * d_model]


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
        return self.head(x)   # [B, 3]


class MagnitudeHead(nn.Module):
    """
    Dual regression head predicting expected TP and SL distances in pips.
    Outputs are constrained positive via Softplus to prevent negative pip deltas.

    Args:
        in_features: Fused manifold width.
        hidden_dim:  Hidden layer dimension.
        dropout:     Dropout probability.
    """
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tp_head = nn.Linear(hidden_dim, 1)   # Take-Profit pips
        self.sl_head = nn.Linear(hidden_dim, 1)   # Stop-Loss pips
        self.act     = nn.Softplus()               # Ensures positive pip outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            mag: [B, 2]  — (tp_pips, sl_pips), always positive
        """
        h  = self.trunk(x)
        tp = self.act(self.tp_head(h))
        sl = self.act(self.sl_head(h))
        return torch.cat([tp, sl], dim=1)  # [B, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class ForexPredictor(nn.Module):
    """
    LemGendary Forex Predictor — Multi-Scale CNN-Transformer Hybrid.

    Architecture:
        1. Per-timeframe TCN encoders (CausalConv1D stacks, 4 layers each)
        2. Cross-timeframe attention (learns which TF is most relevant per bar)
        3. Currency pair embedding (8 pairs, learned)
        4. Dual output heads:
           - DirectionHead:  3-class logits [Down, Sideways, Up]
           - MagnitudeHead:  [TP pips, SL pips]

    Design constraints:
        - Fully stateless (no hidden LSTM state) → safe for ONNX + MT5 EA
        - Causal convolutions only (no future lookahead)
        - ONNX-compatible (no custom ops)
        - Active timeframes are controlled by the Governor's seq_len rung

    Args:
        active_timeframes: List of active timeframe rungs (minutes).
                           Curriculum: start with [1], expand to all 6.
        d_model:           Embedding dimension per timeframe branch.
        n_heads:           Number of cross-timeframe attention heads.
        n_layers:          Number of CausalConv1D layers per encoder.
        head_hidden:       Hidden dim for both output heads.
        dropout:           Global dropout rate.
        in_features:       Features per bar (default: FOREX_FEATURES_PER_BAR).
    """
    def __init__(
        self,
        active_timeframes: list = None,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        head_hidden: int = 256,
        dropout: float = 0.1,
        in_features: int = FOREX_FEATURES_PER_BAR,
    ):
        super().__init__()

        if active_timeframes is None:
            active_timeframes = [60, 240]

        self.active_timeframes = active_timeframes
        self.d_model = d_model

        # Build one TCN encoder per active timeframe
        self.encoders = nn.ModuleDict({
            str(tf): TimeframeEncoder(
                seq_len    = TIMEFRAME_LOOKBACK[tf],
                in_features= in_features,
                d_model    = d_model,
                n_layers   = n_layers,
                dropout    = dropout,
            )
            for tf in active_timeframes
        })

        n_tf = len(active_timeframes)

        # Cross-timeframe attention fuses multiple timeframe embeddings
        self.cross_attn = CrossTimeframeAttention(
            d_model = d_model,
            n_heads = min(n_heads, n_tf),   # Can't exceed n_tf heads
            dropout = dropout,
        )

        # Currency pair embedding → projected into fused manifold
        self.pair_embed   = nn.Embedding(NUM_PAIRS, d_model)
        self.pair_project = nn.Linear(d_model, d_model)

        # Fused manifold dimension: n_tf * d_model (from cross_attn) + d_model (pair embed)
        fused_dim = n_tf * d_model + d_model

        # Manifold projection before heads
        self.fused_project = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(head_hidden),
        )

        self.direction_head = DirectionHead(head_hidden, head_hidden, dropout)
        self.magnitude_head = MagnitudeHead(head_hidden, head_hidden, dropout)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for linear layers, Xavier for embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.pair_embed.weight)

    def forward(
        self,
        tf_inputs: dict,
        pair_idx:  torch.Tensor,
    ) -> dict:
        """
        Args:
            tf_inputs: Dict mapping timeframe (int) → tensor [B, seq_len, features]
                       Only active timeframes need to be present.
            pair_idx:  Long tensor [B] — currency pair index (see PAIR_INDEX).

        Returns:
            Dict with:
                'direction_logits': [B, 3]  — raw logits for CE loss / argmax
                'magnitude':        [B, 2]  — (tp_pips, sl_pips)
        """
        # Encode each active timeframe
        tf_feats = []
        for tf in self.active_timeframes:
            val = tf_inputs.get(tf) if tf in tf_inputs else tf_inputs.get(str(tf))
            if val is None:
                val = next(iter(tf_inputs.values()))
            encoder_key = str(tf)
            if encoder_key in self.encoders:
                tf_feats.append(self.encoders[encoder_key](val))
            else:
                first_enc = next(iter(self.encoders.values()))
                tf_feats.append(first_enc(val))

        # Stack → [B, n_tf, d_model] → cross-timeframe attention → [B, n_tf * d_model]
        tf_stack   = torch.stack(tf_feats, dim=1)
        fused_tf   = self.cross_attn(tf_stack)               # [B, n_tf * d_model]

        # Currency pair embedding
        pair_emb   = self.pair_embed(pair_idx)                # [B, d_model]
        pair_emb   = self.pair_project(pair_emb)              # [B, d_model]

        # Final manifold fusion
        fused      = torch.cat([fused_tf, pair_emb], dim=1)  # [B, n_tf*d_model + d_model]
        manifold   = self.fused_project(fused)                # [B, head_hidden]

        direction  = self.direction_head(manifold)            # [B, 3]
        magnitude  = self.magnitude_head(manifold)            # [B, 2]

        return {
            "direction_logits": direction,
            "magnitude":        magnitude,
        }

    def expand_timeframe(self, new_tf: int):
        """
        Governor hook: add a new timeframe branch without rebuilding the entire model.
        Called by the Governor when advancing to the next curriculum rung.

        Args:
            new_tf: Timeframe in minutes to add (must be in TIMEFRAME_RUNGS).
        """
        if new_tf in self.active_timeframes:
            return
        if new_tf not in TIMEFRAME_LOOKBACK:
            raise ValueError(f"[ForexPredictor] Unknown timeframe: {new_tf}. Valid: {list(TIMEFRAME_LOOKBACK.keys())}")

        key = str(new_tf)
        # Add new encoder branch (inherits same d_model / n_layers)
        ref_encoder = next(iter(self.encoders.values()))
        n_layers    = len(ref_encoder.tcn)
        self.encoders[key] = TimeframeEncoder(
            seq_len     = TIMEFRAME_LOOKBACK[new_tf],
            in_features = ref_encoder.tcn[0].conv.in_channels
                          if hasattr(ref_encoder.tcn[0].conv, 'in_channels') else FOREX_FEATURES_PER_BAR,
            d_model     = self.d_model,
            n_layers    = n_layers,
        )
        self.active_timeframes.append(new_tf)

        # Rebuild cross-attention and projection heads to match new n_tf
        n_tf           = len(self.active_timeframes)
        old_attn       = self.cross_attn
        self.cross_attn = CrossTimeframeAttention(
            d_model = self.d_model,
            n_heads = min(old_attn.attn.num_heads + 1, n_tf),
        )
        fused_dim = n_tf * self.d_model + self.d_model
        old_proj  = self.fused_project
        hidden    = old_proj[0].out_features
        self.fused_project = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Dropout(old_proj[2].p),
            nn.LayerNorm(hidden),
        )
        print(f" [ForexPredictor] Expanded to {n_tf} timeframes. Added: {new_tf}min rung.")

# pylint: disable=duplicate-code
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# [LemGendary Forex Suite v1.0 - SYNC_ID: FOREX_01]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# OHLCV + 5 technical indicators (RSI, MACD, Signal, ATR, BBW)
# + 2 session encoding (sin/cos hour) + 2 volatility regime (ATR percentile, bar range ratio)
# = 14 features per bar
FOREX_FEATURES_PER_BAR = 14

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

# Currency Universe: Titan 4 starting core, extensible up to 16 professional assets
TITAN_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
MAJOR_PAIRS = TITAN_PAIRS

EXTENDED_PAIRS = [
    # G7 Majors (4 Core + 3 G7)
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD",
    "USDCAD", "USDCHF", "AUDUSD", "NZDUSD",
    # High-Beta Crosses
    "EURJPY", "GBPJPY", "EURGBP",
    # Commodities & Energy
    "XAGUSD", "USOIL",
    # Global Equity Indices
    "US500", "USTEC", "GER40"
]

# Currency pair index (0-based) - shared across suite
PAIR_INDEX = {p: i for i, p in enumerate(EXTENDED_PAIRS)}
NUM_PAIRS = len(PAIR_INDEX)


class ForexDualLoss(nn.Module):
    """
    Combined Direction (Classification) + Magnitude (Pip Regression) Loss
    with Anti-Hold Entropy Regularization.
    """
    def __init__(self, mag_weight: float = 0.5, entropy_weight: float = 0.05):
        super().__init__()
        self.mag_weight = mag_weight
        self.entropy_weight = entropy_weight
        # Class weights: Penalize HOLD overconfidence, give higher importance to BUY / SELL
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor([1.2, 0.8, 1.2]))
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, preds: dict, targets: dict) -> torch.Tensor:
        dir_logits = preds["direction"]
        mag_preds = preds["magnitude"]
        
        dir_targets = targets["direction"].to(dir_logits.device)
        mag_targets = targets["magnitude"].to(mag_preds.device)

        loss_dir = self.ce(dir_logits, dir_targets)
        loss_mag = self.smooth_l1(mag_preds, mag_targets)

        # Anti-Hold Entropy Regularization: penalize uniform or collapsed distributions
        probs = F.softmax(dir_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

        return loss_dir + self.mag_weight * loss_mag - self.entropy_weight * entropy


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1DBlock(nn.Module):
    """
    Causal (no future lookahead) dilated Conv1D block with stochastic depth.
    Padding is left-only to ensure strict temporal causality for live trading.
    Stochastic depth randomly skips the block during training (drop-path regularization)
    to prevent the model from memorizing specific temporal patterns.

    Args:
        in_channels:    Input feature depth.
        out_channels:   Output feature depth.
        kernel_size:    Convolution kernel width.
        dilation:       Dilation factor for exponential receptive field growth.
        dropout:        Dropout probability.
        drop_path_rate: Probability of dropping the entire block (stochastic depth).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        # Left-only padding to prevent future data leakage
        self.causal_pad = (kernel_size - 1) * dilation
        self.drop_path_rate = drop_path_rate

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
        # Stochastic depth: randomly drop entire block during training
        if self.training and self.drop_path_rate > 0.0:
            if torch.rand(1).item() < self.drop_path_rate:
                return residual
        x = F.pad(x, (self.causal_pad, 0))       # Causal left-pad only
        x = self.conv(x)                           # [B, out_channels, T]
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

    Args:
        seq_len:     Lookback window length (number of bars).
        in_features: Number of input features per bar (default: FOREX_FEATURES_PER_BAR).
        d_model:     Output embedding dimension.
        n_layers:    Number of CausalConv1D layers.
        dropout:     Dropout probability.
        pair_d_model: Dimension of pair embedding to inject as input bias (0 = disabled).
    """
    def __init__(
        self,
        seq_len: int,
        in_features: int = FOREX_FEATURES_PER_BAR,
        d_model: int = 192,
        n_layers: int = 4,
        dropout: float = 0.1,
        pair_d_model: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pair_d_model = pair_d_model

        # Pair embedding projection: maps [B, d_pair] -> [B, in_features] for additive input bias
        if pair_d_model > 0:
            self.pair_proj = nn.Linear(pair_d_model, in_features)
        else:
            self.pair_proj = None

        layers = []
        in_ch = in_features
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(CausalConv1DBlock(in_ch, d_model, kernel_size=3, dilation=dilation, dropout=dropout))
            in_ch = d_model
        self.tcn = nn.Sequential(*layers)

        # Multi-scale pooling: concat mean + last-step -> 2*d_model
        self.project = nn.Linear(2 * d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)

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
            x = x + pair_bias                                   # [B, seq_len, in_features]
        x = x.transpose(1, 2)          # [B, F, T]
        x = self.tcn(x)                 # [B, d_model, T]

        mean_pool = x.mean(dim=2)       # [B, d_model]  -- macro context
        last_step = x[:, :, -1]         # [B, d_model]  -- most recent state
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
    Outputs are constrained positive via Softplus and clamped to a maximum of 200 pips
    to prevent runaway pip predictions that destabilize the Huber loss gradient signal.

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
        self.tp_head = nn.Linear(hidden_dim, 1)   # Take-Profit pips
        self.sl_head = nn.Linear(hidden_dim, 1)   # Stop-Loss pips
        self.act     = nn.Softplus()               # Ensures positive pip outputs
        self.max_pips = 200.0                      # Hard cap to prevent exploding predictions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            mag: [B, 2]  -- (tp_pips, sl_pips), positive, clamped to max_pips
        """
        h  = self.trunk(x)
        tp = torch.clamp(self.act(self.tp_head(h)), max=self.max_pips)
        sl = torch.clamp(self.act(self.sl_head(h)), max=self.max_pips)
        return torch.cat([tp, sl], dim=1)  # [B, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

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
           - MagnitudeHead:  [TP pips, SL pips] clamped to 200 pip max

    Design constraints:
        - Fully stateless (no hidden LSTM state) -> safe for ONNX + MT5 EA
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
        in_features:       Features per bar (default: FOREX_FEATURES_PER_BAR = 14).
    """
    def __init__(
        self,
        active_timeframes: list | None = None,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        head_hidden: int = 384,
        dropout: float = 0.1,
        in_features: int = FOREX_FEATURES_PER_BAR,
    ):
        super().__init__()

        if active_timeframes is None:
            active_timeframes = [1, 5, 15, 60, 240, 1440]

        self.active_timeframes = active_timeframes
        self.d_model = d_model

        # Pair embedding: early projection for TCN input conditioning
        self.pair_embed   = nn.Embedding(NUM_PAIRS, d_model)
        self.pair_project = nn.Linear(d_model, d_model)

        # Build one TCN encoder per active timeframe
        # Each encoder receives the pair embedding as an additive input bias
        self.encoders = nn.ModuleDict({
            str(tf): TimeframeEncoder(
                seq_len      = TIMEFRAME_LOOKBACK[tf],
                in_features  = in_features,
                d_model      = d_model,
                n_layers     = n_layers,
                dropout      = dropout,
                pair_d_model = d_model,
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
            tf_inputs: Dict mapping timeframe (int) -> tensor [B, seq_len, features]
                       Only active timeframes need to be present.
            pair_idx:  Long tensor [B] -- currency pair index (see PAIR_INDEX).

        Returns:
            Dict with:
                'direction_logits': [B, 3]  -- raw logits for CE loss / argmax
                'magnitude':        [B, 2]  -- (tp_pips, sl_pips)
        """
        # Pair embedding: used both for TCN input conditioning and final manifold fusion
        pair_emb_raw = self.pair_embed(pair_idx)        # [B, d_model]
        pair_emb     = self.pair_project(pair_emb_raw)  # [B, d_model]

        # Encode each active timeframe with pair-conditioned TCN
        tf_feats = []
        for tf in self.active_timeframes:
            val = tf_inputs.get(tf) if tf in tf_inputs else tf_inputs.get(str(tf))
            if val is None:
                val = next(iter(tf_inputs.values()))
            encoder_key = str(tf)
            if encoder_key in self.encoders:
                tf_feats.append(self.encoders[encoder_key](val, pair_emb))
            else:
                first_enc = next(iter(self.encoders.values()))
                tf_feats.append(first_enc(val, pair_emb))

        # Stack -> [B, n_tf, d_model] -> cross-timeframe attention -> [B, n_tf * d_model]
        tf_stack = torch.stack(tf_feats, dim=1)
        fused_tf = self.cross_attn(tf_stack)               # [B, n_tf * d_model]

        # Final manifold fusion with pair embedding
        fused    = torch.cat([fused_tf, pair_emb], dim=1)  # [B, n_tf*d_model + d_model]
        manifold = self.fused_project(fused)                # [B, head_hidden]

        direction = self.direction_head(manifold)            # [B, 3]
        magnitude = self.magnitude_head(manifold)            # [B, 2]

        return {
            "direction":        direction,
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
        n_tf = len(self.active_timeframes)
        # Ensure n_heads divides d_model evenly
        valid_heads = [h for h in [8, 4, 2, 1] if self.d_model % h == 0 and h <= n_tf]
        n_heads = valid_heads[0] if valid_heads else 1
        self.cross_attn = CrossTimeframeAttention(
            d_model = self.d_model,
            n_heads = n_heads,
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

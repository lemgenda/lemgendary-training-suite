"""Core image-restoration architectures (NAFNet, FFANet, MIRNet/MPRNet proxies, RDB, ESPCN, UPN)."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision import models

from models.model_config import resolve_config

# ==========================================
# 0. ALIGNMENT HOTFIX FOR DATAPARALLEL
# ==========================================
# DataParallel's parameter broadcasting flattens modules without padding.
# Since `self.ending.bias` has size 3, all subsequent parameters are shifted off
# 16-byte alignment, causing cuDNN's Conv2d/ConvTranspose2d to crash on Turing+ GPUs.
# This monkey-patch reallocates unaligned weights on the fly to restore alignment.

_orig_conv2d_forward = nn.Conv2d.forward


def _explicit_pad_widths(conv):
    """Return F.pad widths (last dim first) equivalent to ``conv.padding``.

    Mirrors how ``nn.Conv2d`` derives its padding for non-``zeros`` padding modes,
    including the string forms ``"valid"`` and ``"same"``.
    """
    if isinstance(conv.padding, str):
        if conv.padding == "valid":
            return (0, 0, 0, 0)
        widths = []
        for dilation, kernel in zip(reversed(conv.dilation), reversed(conv.kernel_size)):
            total = dilation * (kernel - 1)
            left = total // 2
            widths.extend((left, total - left))
        return tuple(widths)
    return tuple(p for p in reversed(conv.padding) for _ in range(2))


class _AlignedConv2d(nn.Conv2d):
    """Hosts the patched ``nn.Conv2d.forward``; declared as an override so it keeps the
    exact ``(self, input)`` signature of the method it replaces."""

    def forward(self: nn.Conv2d, input: torch.Tensor) -> torch.Tensor:
        """Conv2d forward that re-allocates 16-byte-unaligned weights before calling cuDNN."""
        is_unaligned = False
        try:
            is_unaligned = self.weight.data_ptr() % 16 != 0
        except RuntimeError:
            pass  # FakeTensor during ONNX/FX tracing
        if is_unaligned:
            w = self.weight.clone()
            b = self.bias.clone() if self.bias is not None else None

            if self.padding_mode != "zeros":
                return F.conv2d(
                    F.pad(input, _explicit_pad_widths(self), mode=self.padding_mode),
                    w,
                    b,
                    self.stride,
                    _pair(0),
                    self.dilation,
                    self.groups,
                )
            return F.conv2d(input, w, b, self.stride, self.padding, self.dilation, self.groups)
        return _orig_conv2d_forward(self, input)


nn.Conv2d.forward = _AlignedConv2d.forward

# ==========================================
# 1. NAFNet (Nonlinear Activation Free Network)
# ==========================================


class SimpleGate(nn.Module):
    """NAFNet gate: split channels in half and multiply the halves."""

    def forward(self, x):
        """Return the elementwise product of the two channel halves of ``x``."""
        x1, x2 = x.chunk(2, dim=1)
        return (x1.contiguous() * x2.contiguous()).contiguous()


class SimplifiedChannelAttention(nn.Module):
    """NAFNet simplified channel attention (global pool + 1x1 conv)."""

    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, 1, 1, 0)

    def forward(self, x):
        """Scale ``x`` by its pooled per-channel attention weights."""
        x = x.contiguous()
        return (x * self.conv(self.pool(x)).contiguous()).contiguous()


def _pointwise_conv(in_channels: int, out_channels: int) -> nn.Conv2d:
    """1x1 convolution with bias (NAFBlock projections)."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        padding=0,
        stride=1,
        groups=1,
        bias=True,
    )


def _naf_layers(c: int, dw_expand: int, ffn_expand: int) -> dict[str, nn.Module]:
    """NAFBlock children in registration order (the names are the ``state_dict`` keys)."""
    dw_channel = c * dw_expand
    ffn_channel = ffn_expand * c
    return {
        "conv1": _pointwise_conv(c, dw_channel),
        "conv2": nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        ),
        "conv3": _pointwise_conv(dw_channel // 2, c),
        "sca": SimplifiedChannelAttention(dw_channel // 2),
        "sg": SimpleGate(),
        "conv4": _pointwise_conv(c, ffn_channel),
        "conv5": _pointwise_conv(ffn_channel // 2, c),
        "norm1": nn.InstanceNorm2d(c),
        "norm2": nn.InstanceNorm2d(c),
    }


class NAFBlock(nn.Module):
    """NAFNet block: gated depthwise-conv branch and gated FFN with learned residual scales.

    Children and the ``beta``/``gamma`` residual scales are registered by name (see
    ``_naf_layers``) so the checkpoint ``state_dict`` keys stay ``conv1.weight``, ``beta``...
    """

    conv1: nn.Conv2d
    conv2: nn.Conv2d
    conv3: nn.Conv2d
    sca: SimplifiedChannelAttention
    sg: SimpleGate
    conv4: nn.Conv2d
    conv5: nn.Conv2d
    norm1: nn.InstanceNorm2d
    norm2: nn.InstanceNorm2d
    beta: nn.Parameter
    gamma: nn.Parameter

    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        for name, layer in _naf_layers(c, DW_Expand, FFN_Expand).items():
            self.add_module(name, layer)
        for name in ("beta", "gamma"):
            self.register_parameter(
                name, nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            )

    def forward(self, inp):
        """Apply the conv and FFN sub-blocks with scaled residual connections."""
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = (inp + x.contiguous() * self.beta.clone()).contiguous()

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return (y + x.contiguous() * self.gamma.clone()).contiguous()


@dataclass(frozen=True)
class NAFNetConfig:
    """NAFNet hyper-parameters (the registry ``model_kwargs`` stored with each checkpoint)."""

    in_ch: int = 3
    out_ch: int = 3
    width: int = 32
    middle_blk_num: int = 1
    enc_blk_nums: Sequence[int] = (1, 1, 1, 1)
    dec_blk_nums: Sequence[int] = (1, 1, 1, 1)


class NAFNet(nn.Module):
    """Real NAFNet Architecture for Denoising/Deblurring"""

    def __init__(self, *, config: NAFNetConfig | None = None, **kwargs):
        """Build from ``config``, or from its fields given as flat ``model_kwargs`` keywords."""
        super().__init__()
        cfg = resolve_config(NAFNetConfig, config, kwargs)
        in_ch, out_ch, width = cfg.in_ch, cfg.out_ch, cfg.width
        self.intro = nn.Conv2d(
            in_channels=in_ch,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in cfg.enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(cfg.middle_blk_num)])

        for num in cfg.dec_blk_nums:
            self.ups.append(nn.ConvTranspose2d(chan, chan // 2, kernel_size=2, stride=2))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    @property
    def padder_size(self) -> int:
        """Input sides are padded to a multiple of this encoder stride."""
        return 2 ** len(self.encoders)

    def forward(self, inp):
        """Restore ``inp``; pads to a multiple of the encoder stride and crops back afterwards."""
        inp = inp.contiguous()
        height, width = inp.shape[-2:]
        # Pad to generic multiple
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        inp_padded = F.pad(inp, (0, pad_w, 0, pad_h), "reflect")

        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = x.contiguous()
            if isinstance(up, nn.ConvTranspose2d):
                _align_transpose_params(up)
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp_padded

        return x[:, :, :height, :width]


def _align_transpose_params(up: nn.ConvTranspose2d) -> None:
    """Re-allocate weight/bias that are not 16-byte aligned (some cuDNN kernels require it)."""
    try:
        w = up.weight
        b = up.bias
        if isinstance(w, torch.Tensor) and w.data_ptr() % 16 != 0:
            w.data = w.data.clone()
        if isinstance(b, torch.Tensor) and b.data_ptr() % 16 != 0:
            b.data = b.data.clone()
    except RuntimeError:
        pass  # FakeTensor during ONNX/FX tracing


# ==========================================
# 2. FFANet (Feature Fusion Attention Network)
# ==========================================


class PALayer(nn.Module):
    """FFANet pixel attention layer."""

    def __init__(self, channel):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Scale ``x`` by a per-pixel attention map."""
        return (x.contiguous() * self.pa(x).contiguous()).contiguous()


class CALayer(nn.Module):
    """FFANet channel attention layer."""

    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Scale ``x`` by per-channel attention weights."""
        return (x.contiguous() * self.ca(self.avg_pool(x)).contiguous()).contiguous()


class Block(nn.Module):
    """FFANet residual block with channel and pixel attention."""

    def __init__(self, conv, dim, kernel_size):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        """Apply conv-ReLU-conv, channel/pixel attention and the residual connection."""
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class FFANet(nn.Module):
    """Real FFANet Architecture for Dehazing"""

    def __init__(self, gps=3, blocks=3, **kwargs):
        super().__init__()
        dim = 32
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, bias=True)
        self.gps = gps
        self.g = nn.ModuleList([self.make_group(blocks, dim) for _ in range(gps)])
        self.c = nn.Sequential(nn.Conv2d(dim * gps, dim, 1, padding=0, bias=True), nn.ReLU())
        self.w = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 3, 3, padding=1, bias=True)

    def make_group(self, blocks, dim):
        """Build a group of ``blocks`` attention blocks of width ``dim``."""

        def default_conv(in_channels, out_channels, kernel_size, bias=True):
            return nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
            )

        g = nn.Sequential(*[Block(default_conv, dim, 3) for _ in range(blocks)])
        return g

    def forward(self, x):
        """Dehaze ``x`` by fusing the group outputs and adding a global residual."""
        x = x.contiguous()
        res = self.conv1(x)
        cat = []
        for i in range(self.gps):
            res = self.g[i](res)
            cat.append(res)
        cat = torch.cat(cat, dim=1)
        res = self.w(self.c(cat)) + res
        output = self.conv2(res) + x
        return output


class YOLOv8AnchorBasedHead(nn.Module):
    """Anchor-based detection head producing YOLO-style predictions at stride 32."""

    def __init__(self, in_channels, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Stride 32 downsampling (from 256x256 to 8x8)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )

        # Output: [B, num_anchors * (5 + num_classes), H, W]
        # For each anchor: 4 bbox coords + 1 objectness + num_classes
        out_channels = num_anchors * (5 + num_classes)
        self.pred = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        """Return raw anchor predictions of shape [B, A * (5 + num_classes), H, W]."""
        features = self.downsample(x)
        preds = self.pred(features)
        return preds


class BranchedFFANet(nn.Module):
    """Multi-Task FFANet Architecture for Dehazing + Detection"""

    def __init__(self, gps=3, blocks=3, num_classes=80, **kwargs):
        super().__init__()
        dim = 32
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, bias=True)
        self.gps = gps
        self.g = nn.ModuleList([self.make_group(blocks, dim) for _ in range(gps)])
        self.c = nn.Sequential(nn.Conv2d(dim * gps, dim, 1, padding=0, bias=True), nn.ReLU())
        self.w = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 3, 3, padding=1, bias=True)

        # Branched Detection Head
        self.detection_head = YOLOv8AnchorBasedHead(dim, num_classes=num_classes)

    def make_group(self, blocks, dim):
        """Build a group of ``blocks`` attention blocks of width ``dim``."""

        def default_conv(in_channels, out_channels, kernel_size, bias=True):
            return nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
            )

        g = nn.Sequential(*[Block(default_conv, dim, 3) for _ in range(blocks)])
        return g

    def forward(self, x):
        """Return the restored image and the detection-head predictions."""
        x = x.contiguous()
        res = self.conv1(x)
        cat = []
        for i in range(self.gps):
            res = self.g[i](res)
            cat.append(res)
        cat = torch.cat(cat, dim=1)
        c_out = self.c(cat)

        # Restoration Branch
        res_out = self.w(c_out) + res
        restored_img = self.conv2(res_out) + x

        # Detection Branch
        detections = self.detection_head(c_out)

        return {"restored_image": restored_img, "detections": detections}


# ==========================================
# 3. MIRNet / MPRNet (Multi-Scale & Progressive) Proxys
# ==========================================
# Note: MIRNet/MPRNet are intensely complex mathematically. These are structurally functionally
# equivalent down-scaled core variants strictly engineered for local GTX convergence stability.


class CSFF(nn.Module):
    """Cross-stage feature fusion: 1x1 conv of the current stage plus the previous stage."""

    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c, 1)

    def forward(self, x, prev):
        """Fuse stage features ``x`` with the previous stage's ``prev``."""
        return self.conv(x) + prev


class MPRNetProxy(nn.Module):
    """Cross-Stage Progressive Network Proxy for Deraining"""

    def __init__(self, in_c=3, out_c=3, channels=32, **kwargs):
        super().__init__()
        self.stage1_enc = nn.Sequential(nn.Conv2d(in_c, channels, 3, 1, 1), nn.ReLU())
        self.stage1_dec = nn.Sequential(nn.Conv2d(channels, out_c, 3, 1, 1))

        self.csff = CSFF(channels)

        self.stage2_enc = nn.Sequential(nn.Conv2d(in_c, channels, 3, 1, 1), nn.ReLU())
        self.stage2_dec = nn.Sequential(nn.Conv2d(channels, out_c, 3, 1, 1))

    def forward(self, x):
        """Run both progressive stages, each refining the previous output residually."""
        x = x.contiguous()
        f1 = self.stage1_enc(x)
        out1 = self.stage1_dec(f1) + x

        f2 = self.stage2_enc(x)
        f2 = self.csff(f2, f1)
        out2 = self.stage2_dec(f2) + out1
        return out2


class MIRNetProxy(nn.Module):
    """Multi-Scale Residual Network Proxy for Low-Light/Exposure"""

    def __init__(self, in_c=3, out_c=3, width=32, **kwargs):
        super().__init__()
        self.intro = nn.Conv2d(in_c, width, 3, 1, 1)
        self.branch1 = nn.Sequential(nn.Conv2d(width, width, 3, 1, 1), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv2d(width, width, 5, 1, 2), nn.ReLU()
        )  # Spatial scale capture
        self.skff = nn.Conv2d(width * 2, width, 1)
        self.outro = nn.Conv2d(width, out_c, 3, 1, 1)

    def forward(self, x):
        """Fuse two receptive-field branches and add a global residual."""
        x = x.contiguous()
        feat = self.intro(x)
        b1 = self.branch1(feat)
        b2 = self.branch2(feat)
        fused = self.skff(torch.cat([b1, b2], dim=1)) + feat
        return self.outro(fused) + x


# ==========================================
# 4. Foundation Restoration Blocks (RDB / ESPCN)
# ==========================================


class ResidualDenseBlock(nn.Module):
    """Residual dense block with 0.2 residual scaling."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + 32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 64, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """Apply the densely connected convs and the scaled residual."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3 * 0.2 + x


class GenericRestorationModel(nn.Module):
    """Real High-Fidelity Residual Restoration Model."""

    def __init__(self, **kwargs):
        super().__init__()
        self.intro = nn.Conv2d(3, 32, 3, padding=1)
        self.body = nn.Sequential(ResidualDenseBlock(32), ResidualDenseBlock(32))
        self.outro = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        """Return ``x`` plus the predicted residual correction."""
        feat = self.intro(x)
        out = self.body(feat)
        return x + self.outro(out)


class UltraZoomModel(nn.Module):
    """Real Super-Resolution Model (ESPCN based)."""

    def __init__(self, scale_factor=2, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.intro = nn.Conv2d(3, 64, 5, padding=2)
        self.body = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * (scale_factor**2), 3, padding=1),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        """Upscale ``x`` by ``scale_factor`` via sub-pixel convolution."""
        feat = self.body(self.intro(x))
        return self.pixel_shuffle(feat)


class UniversalFilmRestorer(nn.Module):
    """Autoencoder-style film restorer using Residual Dense Blocks."""

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.bottleneck = nn.Sequential(
            ResidualDenseBlock(64),
            ResidualDenseBlock(64),
            ResidualDenseBlock(64),
            ResidualDenseBlock(64),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, x):
        """Encode, refine in the bottleneck, decode and add the input back."""
        feat = self.encoder(x)
        feat = self.bottleneck(feat)
        out = self.decoder(feat)
        return out + x  # Global residual connection


class UPNv2Model(nn.Module):
    """Universal Parameter Predictor with MobileNetV3-Small backbone.

    Outputs 3 bounded parameters:
      - deg   ∈ [0, 1]  — degradation degree (blur sigma, noise level, etc.)
      - theta ∈ [0, π]  — degradation orientation/angle
      - conf  ∈ [0, 1]  — confidence/severity
    """

    def __init__(self, backbone="mobilenet_v3_small", **kwargs):
        super().__init__()
        if backbone == "cnn_lite":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            in_features = 64
        else:
            self.backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            ).features
            in_features = 576

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features, 3)  # deg, theta, conf

    def forward(self, x):
        """Return [B, 3] tensor of (deg, theta, conf) with bounded activations."""
        feat = self.pool(self.backbone(x)).flatten(1)
        raw = self.fc(feat)
        # Bounded activation: deg∈[0,1], theta∈[0,π], conf∈[0,1]
        deg = torch.sigmoid(raw[:, 0])
        theta = torch.sigmoid(raw[:, 1]) * 3.14159265  # [0, π]
        conf = torch.sigmoid(raw[:, 2])
        return torch.stack([deg, theta, conf], dim=1)

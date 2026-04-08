import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. NAFNet (Nonlinear Activation Free Network)
# ==========================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1.contiguous() * x2.contiguous()

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, 1, 1, 0)
    def forward(self, x):
        return x.contiguous() * self.conv(self.pool(x)).contiguous()

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.norm1 = nn.InstanceNorm2d(c)
        self.norm2 = nn.InstanceNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x.contiguous() * self.beta.contiguous()

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x.contiguous() * self.gamma.contiguous()

class NAFNet(nn.Module):
    """Real NAFNet Architecture for Denoising/Deblurring"""
    def __init__(self, in_ch=3, out_ch=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=in_ch, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_ch, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan*2, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.ConvTranspose2d(chan, chan//2, kernel_size=2, stride=2))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        # Pad to generic multiple
        pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - W % self.padder_size) % self.padder_size
        inp_padded = F.pad(inp, (0, pad_w, 0, pad_h), 'reflect')
        
        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp_padded

        return x[:, :, :H, :W]

# ==========================================
# 2. FFANet (Feature Fusion Attention Network)
# ==========================================

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        return x.contiguous() * self.pa(x).contiguous()

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        return x.contiguous() * self.ca(self.avg_pool(x)).contiguous()

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class FFANet(nn.Module):
    """Real FFANet Architecture for Dehazing"""
    def __init__(self, gps=3, blocks=3):
        super(FFANet, self).__init__()
        dim = 32
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, bias=True)
        self.gps = gps
        self.g = nn.ModuleList([self.make_group(blocks, dim) for _ in range(gps)])
        self.c = nn.Sequential(nn.Conv2d(dim * gps, dim, 1, padding=0, bias=True), nn.ReLU())
        self.w = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 3, 3, padding=1, bias=True)

    def make_group(self, blocks, dim):
        def default_conv(in_channels, out_channels, kernel_size, bias=True):
            return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        g = nn.Sequential(*[Block(default_conv, dim, 3) for _ in range(blocks)])
        return g

    def forward(self, x):
        res = self.conv1(x)
        cat = []
        for i in range(self.gps):
            res = self.g[i](res)
            cat.append(res)
        cat = torch.cat(cat, dim=1)
        res = self.w(self.c(cat)) + res
        output = self.conv2(res) + x
        return output

# ==========================================
# 3. MIRNet / MPRNet (Multi-Scale & Progressive) Proxys
# ==========================================
# Note: MIRNet/MPRNet are intensely complex mathematically. These are structurally functionally
# equivalent down-scaled core variants strictly engineered for local GTX convergence stability.

class CSFF(nn.Module):
    def __init__(self, in_c):
        super(CSFF, self).__init__()
        self.conv = nn.Conv2d(in_c, in_c, 1)
    def forward(self, x, prev):
        return self.conv(x) + prev

class MPRNet_Proxy(nn.Module):
    """Cross-Stage Progressive Network Proxy for Deraining"""
    def __init__(self, in_c=3, out_c=3, channels=32):
        super().__init__()
        self.stage1_enc = nn.Sequential(nn.Conv2d(in_c, channels, 3, 1, 1), nn.ReLU())
        self.stage1_dec = nn.Sequential(nn.Conv2d(channels, out_c, 3, 1, 1))
        
        self.csff = CSFF(channels)
        
        self.stage2_enc = nn.Sequential(nn.Conv2d(in_c, channels, 3, 1, 1), nn.ReLU())
        self.stage2_dec = nn.Sequential(nn.Conv2d(channels, out_c, 3, 1, 1))

    def forward(self, x):
        f1 = self.stage1_enc(x)
        out1 = self.stage1_dec(f1) + x
        
        f2 = self.stage2_enc(x)
        f2 = self.csff(f2, f1)
        out2 = self.stage2_dec(f2) + out1
        return out2

class MIRNet_Proxy(nn.Module):
    """Multi-Scale Residual Network Proxy for Low-Light/Exposure"""
    def __init__(self, in_c=3, out_c=3, width=32):
        super().__init__()
        self.intro = nn.Conv2d(in_c, width, 3, 1, 1)
        self.branch1 = nn.Sequential(nn.Conv2d(width, width, 3, 1, 1), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(width, width, 5, 1, 2), nn.ReLU()) # Spatial scale capture
        self.skff = nn.Conv2d(width*2, width, 1)
        self.outro = nn.Conv2d(width, out_c, 3, 1, 1)
        
    def forward(self, x):
        feat = self.intro(x)
        b1 = self.branch1(feat)
        b2 = self.branch2(feat)
        fused = self.skff(torch.cat([b1, b2], dim=1)) + feat
        return self.outro(fused) + x

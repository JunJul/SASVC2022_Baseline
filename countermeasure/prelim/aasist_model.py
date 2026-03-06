"""
AASIST Model — Self-Contained Implementation
=============================================
Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.

Reference:
    Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
    Graph Attention Networks," ICASSP 2022.
    GitHub: https://github.com/clovaai/aasist

Includes both AASIST (full) and AASIST-L (lightweight) variants.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinc-based front-end (RawNet-style)
class SincConv(nn.Module):
    """
    Sinc-based convolution layer that learns bandpass filters directly
    from raw waveforms. Each filter is parameterized by two learnable
    cutoff frequencies (low_hz, band_hz).
    """
    
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def __init__(self, out_channels, kernel_size, sample_rate=16000,
                 min_low_hz=50, min_band_hz=50):
        super().__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # Initialize filters uniformly in mel scale
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        mel_low = self.to_mel(low_hz)
        mel_high = self.to_mel(high_hz)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = self.to_hz(mel_points)
        
        self.low_hz_ = nn.Parameter(torch.Tensor(hz_points[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_points)).view(-1, 1))
        
        # Hamming window
        n_lin = torch.linspace(0, (kernel_size / 2) - 1, steps=kernel_size // 2)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)
        
        n = (kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate
    
    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : torch.Tensor of shape (batch, 1, time)
        
        Returns
        -------
        torch.Tensor of shape (batch, out_channels, time')
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz, self.sample_rate / 2
        )
        
        band = (high - low)[:, 0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        
        # Compute bandpass filters
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        
        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(
            waveforms, self.filters, stride=1,
            padding=self.kernel_size // 2, bias=None, groups=1
        )


# Graph Attention Layer
class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer."""
    
    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.temperature = temperature
    
    def edge_attention(self, h):
        """Compute attention scores between all node pairs."""
        B, N, D = h.shape
        
        # All pairs
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        
        edge_input = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, 2D)
        attn = self.attn_fc(edge_input).squeeze(-1)  # (B, N, N)
        attn = F.leaky_relu(attn, negative_slope=0.2)
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=-1)
        
        return attn
    
    def forward(self, h):
        """
        Parameters
        ----------
        h : torch.Tensor of shape (batch, num_nodes, in_dim)
        
        Returns
        -------
        torch.Tensor of shape (batch, num_nodes, out_dim)
        """
        h = self.fc(h)
        attn = self.edge_attention(h)
        out = torch.bmm(attn, h)
        return F.elu(out)


# AASIST Encoder Blocks
class ResBlock(nn.Module):
    """Residual block with 1D convolutions."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        
        self.downsample = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
    
    def forward(self, x):
        identity = self.downsample(x)
        out = F.selu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.selu(out + identity)


class Encoder(nn.Module):
    """Stack of ResBlocks with adaptive pooling for graph construction."""
    
    def __init__(self, filts):
        super().__init__()
        blocks = []
        for i in range(len(filts) - 1):
            in_ch = filts[i] if isinstance(filts[i], int) else filts[i][-1]
            out_ch = filts[i+1] if isinstance(filts[i+1], int) else filts[i+1][-1]
            blocks.append(ResBlock(in_ch, out_ch))
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        """Returns list of feature maps at each resolution."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = F.adaptive_avg_pool1d(x, x.shape[-1] // 2)
        return features


#main model 
class AASIST(nn.Module):
    """
    Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.
    
    Parameters
    ----------
    config : dict
        Model configuration. Use AASIST_CONFIG or AASIST_L_CONFIG from config.py.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.nb_samp = config["nb_samp"]
        first_conv = config["first_conv"]
        filts = config["filts"]
        gat_dims = config["gat_dims"]
        pool_ratios = config["pool_ratios"]
        temperatures = config["temperatures"]
        
        # Sinc convolution
        self.sinc_conv = SincConv(
            out_channels=first_conv,
            kernel_size=129,
            sample_rate=16000,
        )
        self.sinc_bn = nn.BatchNorm1d(first_conv)
        
        # Spectral encoder 
        self.spec_encoder = Encoder([first_conv] + [f[-1] if isinstance(f, list) else f for f in filts])
        
        # Temporal encoder
        self.temp_encoder = Encoder([first_conv] + [f[-1] if isinstance(f, list) else f for f in filts])
        
        # Graph attention 
        final_dim = filts[-1][-1] if isinstance(filts[-1], list) else filts[-1]
        self.gat1 = GraphAttentionLayer(final_dim, gat_dims[0], temperatures[0])
        self.gat2 = GraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[1])
        
        # Readout & classifier 
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(gat_dims[1] * 2, gat_dims[1]),
            nn.SELU(),
            nn.Linear(gat_dims[1], 2),
        )
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, nb_samp)
            Raw waveform input.
        
        Returns
        -------
        torch.Tensor of shape (batch, 2)
            Logits for [spoof, bonafide].
        """
        # Add channel dimension: (B, T) → (B, 1, T)
        x = x.unsqueeze(1)
        
        # Sinc front-end
        x = self.sinc_conv(x)
        x = F.selu(self.sinc_bn(x))
        x = F.max_pool1d(torch.abs(x), 3)
        
        # Dual-branch encoding
        spec_feats = self.spec_encoder(x)
        temp_feats = self.temp_encoder(x)
        
        # Take final-level features from each branch
        spec_out = self.pool(spec_feats[-1]).squeeze(-1)  # (B, D)
        temp_out = self.pool(temp_feats[-1]).squeeze(-1)  # (B, D)
        
        # Construct graph nodes: stack spectral + temporal
        # (B, 2, D) — 2 nodes per sample
        nodes = torch.stack([spec_out, temp_out], dim=1)
        
        # Graph attention
        nodes = self.gat1(nodes)
        nodes = self.gat2(nodes)
        
        # Readout: concatenate both node embeddings
        out = nodes.reshape(nodes.size(0), -1)  # (B, 2*gat_dims[-1])
        
        # Classify
        logits = self.classifier(out)
        
        return logits


def build_model(variant="AASIST"):
    """
    Build an AASIST model.
    
    Parameters
    ----------
    variant : str
        "AASIST" for full model, "AASIST-L" for lightweight.
    
    Returns
    -------
    AASIST model instance.
    """
    from config import AASIST_CONFIG, AASIST_L_CONFIG
    
    config = AASIST_CONFIG if variant == "AASIST" else AASIST_L_CONFIG
    model = AASIST(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n  Model: {variant}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    """verify model accepts correct input shape."""

    
    for variant in ["AASIST", "AASIST-L"]:
        model = build_model(variant)
        model.eval()
        
        # Simulate a batch of raw waveforms
        batch_size = 4
        dummy_input = torch.randn(batch_size, 64600)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output (logits): {output[0].tolist()}")
        
        # Verify output is (batch, 2)
        assert output.shape == (batch_size, 2), \
            f"Expected ({batch_size}, 2), got {output.shape}"
        print(f"  {variant} test passed!")

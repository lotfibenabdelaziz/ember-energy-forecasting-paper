"""
src/model.py — Deep Learning Model Architectures
=================================================
Ember Energy | IEEE Paper — Deep Learning module

4 architectures, mirrors notebook 04_deeplearning_enhanced_patched.ipynb exactly:
    - MLPForecaster    : flat MLP with BatchNorm + Dropout
    - TCNForecaster    : dilated causal convolutions + residuals
    - NBeatsForecaster : trend (polynomial) + generic stack
    - TFTForecaster    : lightweight Temporal Fusion Transformer

Each model: input [B, seq_len, n_features] → output [B, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# Model 1 — MLP (Multi-Layer Perceptron)
# ═══════════════════════════════════════════════════════════════════════════════

class MLPForecaster(nn.Module):
    """
    Flat MLP: flattens the (seq_len, n_feat) window and passes through
    fully-connected layers with BatchNorm + Dropout regularisation.

    Architecture:
        [seq_len x n_features] → Flatten
          → FC(256) → BN → ReLU → Dropout(0.3)
          → FC(128) → BN → ReLU → Dropout(0.2)
          → FC(64)  → BN → ReLU
          → FC(1)   → Demand
    """

    def __init__(
        self,
        seq_len:    int,
        n_features: int,
        hidden:     tuple[int, ...] = (256, 128, 64),
        dropout:    tuple[float, ...] = (0.3, 0.2),
    ) -> None:
        super().__init__()
        in_dim = seq_len * n_features
        layers: list[nn.Module] = []
        prev  = in_dim
        drops = list(dropout) + [0.0] * len(hidden)

        for i, h in enumerate(hidden):
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ]
            if drops[i] > 0:
                layers.append(nn.Dropout(drops[i]))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, n_features]
        return self.net(x.flatten(1))


# ═══════════════════════════════════════════════════════════════════════════════
# Model 2 — TCN (Temporal Convolutional Network)
# ═══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """Conv1d with left-padding to ensure causal (no future leakage)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Two causal convs + residual connection."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.drop(out)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    """
    Stacked TCN blocks with exponentially increasing dilation.
    Takes last timestep output -> FC(1).

    With dilations=(1,2,4) and kernel_size=3:
        receptive field = (3-1) * (1+2+4) + 1 = 15 steps
    """

    def __init__(
        self,
        n_features:  int,
        n_channels:  int = 64,
        kernel_size: int = 3,
        dilations:   tuple[int, ...] = (1, 2, 4),
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for d in dilations:
            layers.append(TCNBlock(in_ch, n_channels, kernel_size, d, dropout))
            in_ch = n_channels
        self.tcn    = nn.Sequential(*layers)
        self.linear = nn.Linear(n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, n_features] -> transpose -> [B, n_features, seq_len]
        x = x.transpose(1, 2)
        x = self.tcn(x)        # [B, n_channels, seq_len]
        x = x[:, :, -1]        # last timestep
        return self.linear(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Model 3 — N-BEATS (Neural Basis Expansion Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

class NBeatsBlock(nn.Module):
    """
    Single N-BEATS block.
    FC stack -> (theta_backcast, theta_forecast) -> basis expansion.

    basis_type:
        "trend"   : polynomial basis [1, t, t^2, t^3, ...]
        "generic" : learned/interpolated basis
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        horizon:    int,
        n_layers:   int = 4,
        hidden:     int = 128,
        basis_type: str = "generic",
    ) -> None:
        super().__init__()
        self.horizon    = horizon
        self.basis_type = basis_type
        self.theta_size = theta_size

        fc_layers: list[nn.Module] = []
        prev = input_size
        for _ in range(n_layers):
            fc_layers += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        self.fc = nn.Sequential(*fc_layers)

        self.theta_b = nn.Linear(hidden, theta_size, bias=False)
        self.theta_f = nn.Linear(hidden, theta_size, bias=False)

    def basis_expansion(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Project theta through polynomial (trend) or identity (generic) basis."""
        if self.basis_type == "trend":
            p = torch.arange(self.theta_size, dtype=torch.float32, device=theta.device)
            T = t.unsqueeze(-1) ** p.unsqueeze(0)   # [len(t), theta_size]
            return (
                torch.einsum("bt,Tt->bT", theta, T)
                if theta.dim() == 2
                else theta @ T.t()
            )

        # Generic: learned linear mapping back to time axis
        L = len(t)
        if self.theta_size == L:
            return theta
        return F.interpolate(
            theta.unsqueeze(1), size=L, mode="linear", align_corners=False
        ).squeeze(1)

    def forward(
        self, x: torch.Tensor, backcast_t: torch.Tensor, forecast_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h       = self.fc(x)
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        backcast = self.basis_expansion(theta_b, backcast_t)
        forecast = self.basis_expansion(theta_f, forecast_t)
        return backcast, forecast


class NBeatsForecaster(nn.Module):
    """
    N-BEATS with two stacks:
        Stack 1 — Trend block   (polynomial basis, degree 3)
        Stack 2 — Generic block (learned basis)
    Residuals passed between stacks.

    Operates on the flattened window (all features), but the residual
    subtraction is applied only to the TARGET (column 0) sub-sequence.
    """

    def __init__(
        self,
        seq_len:      int,
        horizon:      int = 1,
        n_features:   int = 1,
        trend_degree: int = 3,
        hidden:       int = 128,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon

        in_size = seq_len * n_features

        # Trend stack
        self.trend_block = NBeatsBlock(
            in_size, theta_size=trend_degree + 1, horizon=horizon,
            n_layers=4, hidden=hidden, basis_type="trend",
        )
        # Generic stack
        self.generic_block = NBeatsBlock(
            in_size, theta_size=max(horizon, 4), horizon=horizon,
            n_layers=4, hidden=hidden, basis_type="generic",
        )

        # Time grids
        self.register_buffer("backcast_t", torch.linspace(-1, 1, seq_len))
        self.register_buffer("forecast_t", torch.linspace(0, 1, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, n_features]
        x_flat = x.flatten(1)   # [B, seq_len * n_features]

        # Stack 1: Trend
        backcast1, forecast1 = self.trend_block(x_flat, self.backcast_t, self.forecast_t)

        # Residual — subtract backcast from the target (column 0) sub-sequence
        demand_seq = x[:, :, 0]            # [B, seq_len]
        residual   = demand_seq - backcast1
        x_res      = x_flat.clone()
        x_res[:, : self.seq_len] = residual

        # Stack 2: Generic (on residual)
        _, forecast2 = self.generic_block(x_res, self.backcast_t, self.forecast_t)

        # Sum forecasts -> [B, horizon, 1]
        return (forecast1 + forecast2).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Model 4 — TFT (Temporal Fusion Transformer) — lightweight
# ═══════════════════════════════════════════════════════════════════════════════

class GatedResidualNetwork(nn.Module):
    """GRN: FC -> ELU -> FC -> GLU gate -> residual + LayerNorm."""

    def __init__(
        self,
        d_in:        int,
        d_hidden:    int,
        d_out:       int,
        dropout:     float = 0.1,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.fc1  = nn.Linear(d_in + (context_dim or 0), d_hidden)
        self.fc2  = nn.Linear(d_hidden, d_out * 2)   # *2 for GLU
        self.gate = nn.GLU(dim=-1)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = x if context is None else torch.cat([x, context], dim=-1)
        h = F.elu(self.fc1(h))
        h = self.drop(self.gate(self.fc2(h)))
        return self.norm(h + self.skip(x))


class VariableSelectionNetwork(nn.Module):
    """Soft feature selection: learned weights per feature per timestep."""

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout) for _ in range(n_features)
        ])
        self.weight  = GatedResidualNetwork(n_features, d_model, n_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, n_features]
        _, _, num_feat = x.shape

        feat_emb = torch.stack(
            [self.grns[i](x[..., i : i + 1]) for i in range(num_feat)], dim=-2
        )   # [B, T, F, d_model]

        weights = self.softmax(self.weight(x))   # [B, T, F]
        out = (feat_emb * weights.unsqueeze(-1)).sum(dim=-2)   # [B, T, d_model]
        return out, weights


class TFTForecaster(nn.Module):
    """
    Lightweight TFT for short annual panel data.

    Encoder LSTM on past window -> multi-head self-attention -> GRN -> FC(1).
    Variable Selection Network provides per-feature importance weights.
    """

    def __init__(
        self,
        n_features: int,
        d_model:    int = 32,
        n_heads:    int = 2,
        seq_len:    int = 5,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.vsn  = VariableSelectionNetwork(n_features, d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True, dropout=0.0)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.grn  = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self._weights: torch.Tensor | None = None  # stored attention weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, n_features]
        x_sel, _ = self.vsn(x)                       # [B, T, d_model]
        enc, _   = self.lstm(x_sel)                  # [B, T, d_model]
        attn_out, weights = self.attn(enc, enc, enc, need_weights=True)
        self._weights = weights.detach().cpu()       # store for interpretability
        out = self.grn(self.norm(attn_out + enc))
        return self.head(out[:, -1, :])              # last timestep

    def get_attention_weights(self) -> torch.Tensor | None:
        return self._weights


# ═══════════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════════

def build_model_registry(seq_len: int, n_features: int) -> dict[str, dict]:
    """
    Build the MODEL_REGISTRY dict — mirrors notebook exactly.

    Each entry: {"cls": ModelClass, "kwargs": {...constructor args...}}
    """
    return {
        "MLP": {
            "cls": MLPForecaster,
            "kwargs": {
                "seq_len": seq_len, "n_features": n_features,
                "hidden": (256, 128, 64), "dropout": (0.3, 0.2),
            },
        },
        "TCN": {
            "cls": TCNForecaster,
            "kwargs": {
                "n_features": n_features, "n_channels": 64,
                "kernel_size": 3, "dilations": (1, 2, 4), "dropout": 0.2,
            },
        },
        "N-BEATS": {
            "cls": NBeatsForecaster,
            "kwargs": {
                "seq_len": seq_len, "horizon": 1, "n_features": n_features,
                "trend_degree": 3, "hidden": 128,
            },
        },
        "TFT": {
            "cls": TFTForecaster,
            "kwargs": {
                "n_features": n_features, "d_model": 32, "n_heads": 2,
                "seq_len": seq_len, "dropout": 0.1,
            },
        },
    }

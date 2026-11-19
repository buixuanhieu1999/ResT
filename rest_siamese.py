"""Faithful, configurable reconstruction of the ResT Siamese network.

Based on Figures 1-4 and Sections 3.2-3.6 of:
"Writer-Independent Offline Signature Verification Using Local and Global
Feature Fusion", Symmetry 2026, 18, 149.

The paper specifies the major modules but not every layer hyperparameter. The
defaults below use a ViT-B/16-style global branch and a standard ResNet-18 local
trunk; see README_rest_reconstruction.md for the explicit assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ResTConfig:
    """Network configuration matching the paper where values are disclosed."""

    image_size: int = 224
    in_channels: int = 3

    # The paper says ImageNet-21k-pretrained ViT but omits the exact variant.
    # ViT-B/16 is used as the reconstruction default.
    patch_size: int = 16
    vit_dim: int = 768
    vit_depth: int = 12
    vit_heads: int = 12
    vit_mlp_ratio: float = 4.0

    # A standard ResNet-18 with base width 64 produces the stated 512-D local
    # feature vector.
    local_base_channels: int = 64

    dropout: float = 0.0
    attention_dropout: float = 0.0
    normalize_fused_embedding: bool = False
    gradient_checkpointing: bool = False


class SharedConvPreprocessor(nn.Module):
    """Convolution shown before both branches in Figure 2."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SPDConv(nn.Module):
    """Spatial-to-depth followed by a non-strided convolution (paper Eq. 1)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(
                in_channels * 4,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if height % 2 or width % 2:
            raise ValueError(
                f"SPDConv requires even spatial dimensions, received {height}x{width}."
            )
        x = torch.cat(
            (
                x[..., 0::2, 0::2],
                x[..., 1::2, 0::2],
                x[..., 0::2, 1::2],
                x[..., 1::2, 1::2],
            ),
            dim=1,
        )
        return self.refine(x)


class SpatialReconstructionUnit(nn.Module):
    """SRU half of SCConv: separate, gate, and cross-reconstruct spatial maps."""

    def __init__(
        self, channels: int, group_count: int = 16, gate_threshold: float = 0.5
    ) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("SRU requires an even channel count for reconstruction.")
        if channels % group_count:
            raise ValueError("SRU channels must be divisible by group_count.")
        self.norm = nn.GroupNorm(group_count, channels)
        self.gate_threshold = gate_threshold

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm(x)
        gamma = self.norm.weight
        denominator = gamma.sum()
        denominator = torch.where(
            denominator.abs() < 1e-12,
            torch.ones_like(denominator),
            denominator,
        )
        gamma_weights = (gamma / denominator).view(1, -1, 1, 1)
        importance = torch.sigmoid(normalized * gamma_weights)

        informative = torch.where(
            importance >= self.gate_threshold,
            torch.ones_like(importance),
            importance,
        )
        less_informative = torch.where(
            importance >= self.gate_threshold,
            torch.zeros_like(importance),
            importance,
        )

        x_info = informative * x
        x_less = less_informative * x
        x11, x12 = torch.chunk(x_info, 2, dim=1)
        x21, x22 = torch.chunk(x_less, 2, dim=1)
        return torch.cat((x11 + x22, x12 + x21), dim=1)


class ChannelReconstructionUnit(nn.Module):
    """CRU half of SCConv: split, transform, and adaptively fuse channels."""

    def __init__(
        self,
        channels: int,
        alpha: float = 0.5,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("CRU alpha must lie strictly between 0 and 1.")

        upper_channels = int(alpha * channels)
        lower_channels = channels - upper_channels
        upper_squeezed = upper_channels // squeeze_ratio
        lower_squeezed = lower_channels // squeeze_ratio

        if min(upper_squeezed, lower_squeezed) < 1:
            raise ValueError("CRU squeeze_ratio is too large for the channel count.")
        if upper_squeezed % group_size or channels % group_size:
            raise ValueError(
                "CRU upper squeezed channels and output channels must be divisible "
                "by group_size."
            )

        self.upper_channels = upper_channels
        self.lower_channels = lower_channels
        self.squeeze_upper = nn.Conv2d(
            upper_channels, upper_squeezed, kernel_size=1, bias=False
        )
        self.squeeze_lower = nn.Conv2d(
            lower_channels, lower_squeezed, kernel_size=1, bias=False
        )
        self.group_conv = nn.Conv2d(
            upper_squeezed,
            channels,
            kernel_size=group_kernel_size,
            stride=1,
            padding=group_kernel_size // 2,
            groups=group_size,
            bias=False,
        )
        self.upper_pointwise = nn.Conv2d(
            upper_squeezed, channels, kernel_size=1, bias=False
        )
        self.lower_pointwise = nn.Conv2d(
            lower_squeezed,
            channels - lower_squeezed,
            kernel_size=1,
            bias=False,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        upper, lower = torch.split(
            x, (self.upper_channels, self.lower_channels), dim=1
        )
        upper = self.squeeze_upper(upper)
        lower = self.squeeze_lower(lower)

        y1 = self.group_conv(upper) + self.upper_pointwise(upper)
        y2 = torch.cat((self.lower_pointwise(lower), lower), dim=1)
        fused = torch.cat((y1, y2), dim=1)
        fused = F.softmax(self.pool(fused), dim=1) * fused
        y1, y2 = torch.chunk(fused, 2, dim=1)
        return y1 + y2


class SCConv(nn.Module):
    """Spatial and Channel Reconstruction Convolution (SRU followed by CRU)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        group_count = min(16, channels)
        while channels % group_count:
            group_count -= 1
        self.sru = SpatialReconstructionUnit(
            channels, group_count=group_count, gate_threshold=0.5
        )
        self.cru = ChannelReconstructionUnit(
            channels,
            alpha=0.5,  # explicitly specified in the paper
            squeeze_ratio=2,
            group_size=2,
            group_kernel_size=3,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.cru(self.sru(x))


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Standard ResNet-18 basic residual block."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.activation(out + identity)


class ResNet18Trunk(nn.Module):
    """Four ResNet-18 stages; the SPD module replaces strided stem pooling."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.in_channels = base_channels
        self.layer1 = self._make_stage(base_channels, blocks=2, stride=1)
        self.layer2 = self._make_stage(base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_stage(base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_stage(base_channels * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = base_channels * 8

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(
            BasicBlock(self.in_channels, out_channels) for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.pool(x), 1)


class LocalBranch(nn.Module):
    """SPD-Conv -> SCConv -> ResNet-18 -> 512-D local feature."""

    def __init__(self, in_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        self.spd = SPDConv(in_channels, base_channels)
        self.scconv = SCConv(base_channels)
        self.resnet = ResNet18Trunk(base_channels)
        self.output_dim = self.resnet.output_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet(self.scconv(self.spd(x)))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm attention and MLP block shown in Figure 2."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = x + self.dropout(attended)
        return x + self.mlp(self.norm2(x))


class VisionTransformerBranch(nn.Module):
    """Patch embedding plus Transformer encoder for the global feature vector."""

    def __init__(self, config: ResTConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size:
            raise ValueError("image_size must be divisible by patch_size.")
        if config.vit_dim % config.vit_heads:
            raise ValueError("vit_dim must be divisible by vit_heads.")

        patches_per_side = config.image_size // config.patch_size
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.vit_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        token_count = patches_per_side * patches_per_side + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.vit_dim))
        self.position = nn.Parameter(torch.zeros(1, token_count, config.vit_dim))
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.vit_dim,
                config.vit_heads,
                config.vit_mlp_ratio,
                config.dropout,
                config.attention_dropout,
            )
            for _ in range(config.vit_depth)
        )
        self.norm = nn.LayerNorm(config.vit_dim)
        self.output_dim = config.vit_dim
        self.gradient_checkpointing = config.gradient_checkpointing

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.embedding_dropout(x + self.position)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)[:, 0]


class ResTEncoder(nn.Module):
    """Parallel global/local encoder with feature concatenation."""

    def __init__(self, config: ResTConfig = ResTConfig()) -> None:
        super().__init__()
        self.config = config
        self.shared_conv = SharedConvPreprocessor(config.in_channels)
        self.global_branch = VisionTransformerBranch(config)
        self.local_branch = LocalBranch(
            config.in_channels, base_channels=config.local_base_channels
        )
        self.output_dim = self.global_branch.output_dim + self.local_branch.output_dim

    def forward(
        self, x: Tensor, return_parts: bool = False
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        expected = (self.config.image_size, self.config.image_size)
        if tuple(x.shape[-2:]) != expected:
            raise ValueError(
                f"ResT expects {expected[0]}x{expected[1]} inputs, received "
                f"{x.shape[-2]}x{x.shape[-1]}."
            )

        convolved = self.shared_conv(x)
        # Figure 2 shows an input residual around the convolution before patching.
        global_feature = self.global_branch(x + convolved)
        local_feature = self.local_branch(convolved)
        fused = torch.cat((local_feature, global_feature), dim=1)
        if self.config.normalize_fused_embedding:
            fused = F.normalize(fused, p=2, dim=1)

        if return_parts:
            return fused, {"local": local_feature, "global": global_feature}
        return fused


class SiameseResT(nn.Module):
    """Weight-sharing pair network from Figure 1."""

    def __init__(self, encoder: ResTEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else ResTEncoder()

    def forward(self, reference: Tensor, query: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        reference_feature = self.encoder(reference)
        query_feature = self.encoder(query)
        distance = torch.linalg.vector_norm(
            reference_feature - query_feature, ord=2, dim=1
        )
        return reference_feature, query_feature, distance

    @torch.no_grad()
    def verify(self, reference: Tensor, query: Tensor, threshold: float) -> Tensor:
        """Return True for genuine/same-writer pairs using paper Eq. 8."""
        _, _, distance = self(reference, query)
        return distance <= threshold


class ContrastiveLoss(nn.Module):
    """Paper Eq. 7. Labels use 1=same type and 0=different type."""

    def __init__(
        self,
        margin: float = 1.0,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, distance: Tensor, same_type_label: Tensor) -> Tensor:
        label = same_type_label.to(dtype=distance.dtype)
        positive = self.positive_weight * label * distance.square()
        negative = (
            self.negative_weight
            * (1.0 - label)
            * F.relu(self.margin - distance).square()
        )
        return (positive + negative).mean()


def tiny_config_for_smoke_test() -> ResTConfig:
    """Small configuration used only to validate tensor flow quickly on CPU."""

    return ResTConfig(
        image_size=64,
        patch_size=8,
        vit_dim=96,
        vit_depth=2,
        vit_heads=4,
        local_base_channels=16,
    )


if __name__ == "__main__":
    torch.manual_seed(7)
    config = tiny_config_for_smoke_test()
    network = SiameseResT(ResTEncoder(config)).eval()
    reference = torch.randn(2, config.in_channels, config.image_size, config.image_size)
    query = torch.randn_like(reference)
    with torch.no_grad():
        reference_feature, query_feature, distance = network(reference, query)
    assert reference_feature.shape == (2, network.encoder.output_dim)
    assert query_feature.shape == reference_feature.shape
    assert distance.shape == (2,)
    print(
        "Smoke test passed:",
        f"embedding={tuple(reference_feature.shape)}",
        f"distance={tuple(distance.shape)}",
    )

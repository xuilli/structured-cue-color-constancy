from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import EPCCFeatureExtractor


def _validate_16d_features(features: torch.Tensor) -> None:
    if features.ndim != 2 or features.shape[1] != 16:
        raise ValueError(
            "Structured interaction experiments require [B, 16] ePCC features, "
            f"got {tuple(features.shape)}"
        )


def _validate_regressor_features(
    features: torch.Tensor, input_dim: int, regressor_name: str
) -> None:
    if features.ndim != 2 or features.shape[1] != input_dim:
        raise ValueError(
            f"{regressor_name} expects [B, {input_dim}] features, "
            f"got {tuple(features.shape)}"
        )


def get_activation(
    name: str, negative_slope: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "relu":
        return F.relu
    if name == "leaky_relu":
        return lambda inputs: F.leaky_relu(inputs, negative_slope=negative_slope)
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported FastKAN base activation: {name}")


class SplineLinear(nn.Linear):
    def __init__(
        self, input_dim: int, output_dim: int, init_scale: float = 0.1
    ) -> None:
        self.init_scale = init_scale
        super().__init__(input_dim, output_dim, bias=False)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
        self, grid_min: float = -2.0, grid_max: float = 2.0, num_centers: int = 8
    ) -> None:
        super().__init__()
        if num_centers < 2:
            raise ValueError("RBF num_centers must be at least 2")
        if grid_max <= grid_min:
            raise ValueError("grid_max must be greater than grid_min")
        self.output_dim = num_centers
        self.register_buffer("grid", torch.linspace(grid_min, grid_max, num_centers))
        self.register_buffer(
            "denominator", torch.tensor((grid_max - grid_min) / (num_centers - 1))
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(((inputs[..., None] - self.grid) / self.denominator) ** 2))


class BSplineBasis(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        grid_intervals: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()

        if grid_intervals < 1:
            raise ValueError("bspline_grid_intervals must be at least 1")
        if spline_order < 0:
            raise ValueError("bspline_order cannot be negative")
        if grid_max <= grid_min:
            raise ValueError("grid_max must be greater than grid_min")

        grid_step = (grid_max - grid_min) / grid_intervals

        knots = (
            torch.arange(-spline_order, grid_intervals + spline_order + 1) * grid_step
            + grid_min
        )

        self.spline_order = spline_order
        self.grid_intervals = grid_intervals
        self.output_dim = grid_intervals + spline_order

        self.register_buffer("knots", knots)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expanded = inputs.unsqueeze(-1)

        basis = ((expanded >= self.knots[:-1]) & (expanded < self.knots[1:])).to(
            dtype=inputs.dtype
        )

        for degree in range(1, self.spline_order + 1):
            left_knots = self.knots[: -(degree + 1)]
            left_upper = self.knots[degree:-1]
            right_lower = self.knots[1:-degree]
            right_knots = self.knots[degree + 1 :]

            left = ((expanded - left_knots) / (left_upper - left_knots)) * basis[
                ..., :-1
            ]

            right = ((right_knots - expanded) / (right_knots - right_lower)) * basis[
                ..., 1:
            ]

            basis = left + right

        return basis.contiguous()


class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float,
        grid_max: float,
        rbf_num_centers: int,
        bspline_grid_intervals: int,
        bspline_order: int,
        basis_type: str,
        use_base_update: bool,
        use_layernorm: bool,
        base_activation: Callable[[torch.Tensor], torch.Tensor],
        spline_weight_init_scale: float,
    ) -> None:
        super().__init__()

        self.layernorm = nn.LayerNorm(input_dim) if use_layernorm else None

        if basis_type == "rbf":
            self.basis = RadialBasisFunction(
                grid_min=grid_min, grid_max=grid_max, num_centers=rbf_num_centers
            )
        elif basis_type == "b_spline":
            self.basis = BSplineBasis(
                grid_min=grid_min,
                grid_max=grid_max,
                grid_intervals=bspline_grid_intervals,
                spline_order=bspline_order,
            )
        else:
            raise ValueError(
                "fastkan_basis must be 'rbf' or 'b_spline', " f"got {basis_type!r}"
            )

        self.spline_linear = SplineLinear(
            input_dim * self.basis.output_dim, output_dim, spline_weight_init_scale
        )
        self.base_activation = base_activation
        self.base_linear = nn.Linear(input_dim, output_dim) if use_base_update else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.layernorm(inputs) if self.layernorm is not None else inputs

        basis = self.basis(normalized)
        spline = self.spline_linear(basis.flatten(start_dim=-2))

        if self.base_linear is None:
            return spline

        return spline + self.base_linear(self.base_activation(inputs))


class FastKANRegressor(nn.Module):
    def __init__(self, cfg, input_dim: int) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("FastKAN input_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dims = cfg.validated_hidden_dims()
        basis_type = cfg.fastkan_basis.lower().strip()
        if basis_type not in {"b_spline", "rbf"}:
            raise ValueError(
                "fastkan_basis must be 'b_spline' or 'rbf', "
                f"got {cfg.fastkan_basis!r}"
            )

        base_activation = get_activation(
            cfg.fastkan_base_activation, cfg.leaky_relu_slope
        )

        def make_layer(
            layer_input_dim: int, layer_output_dim: int, use_layernorm: bool = True
        ) -> FastKANLayer:
            return FastKANLayer(
                input_dim=layer_input_dim,
                output_dim=layer_output_dim,
                grid_min=cfg.fastkan_grid_min,
                grid_max=cfg.fastkan_grid_max,
                rbf_num_centers=cfg.rbf_num_centers,
                bspline_grid_intervals=cfg.bspline_grid_intervals,
                bspline_order=cfg.bspline_order,
                basis_type=basis_type,
                use_base_update=cfg.fastkan_use_base_update,
                use_layernorm=(use_layernorm and cfg.fastkan_use_layernorm),
                base_activation=base_activation,
                spline_weight_init_scale=cfg.fastkan_spline_weight_init_scale,
            )

        layer_dims = [input_dim, *self.hidden_dims, 2]
        self.layers = nn.ModuleList(
            [
                make_layer(layer_input_dim, layer_output_dim)
                for layer_input_dim, layer_output_dim in zip(
                    layer_dims[:-1], layer_dims[1:]
                )
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_regressor_features(features, self.input_dim, "FastKAN")

        hidden = features
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class StructuredTokenEncoder(nn.Module):
    def __init__(self, token_dim: int, negative_slope: float) -> None:
        super().__init__()

        if token_dim <= 0:
            raise ValueError("token_dim must be positive")

        self.token_dim = token_dim
        self.negative_slope = negative_slope

        self.content_projection = nn.Linear(2, token_dim)
        self.statistic_embedding = nn.Embedding(4, token_dim)
        self.source_embedding = nn.Embedding(2, token_dim)

        self.register_buffer(
            "statistic_ids",
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "source_ids",
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
            persistent=False,
        )

        self._initialize_parameters(negative_slope)

    def _initialize_parameters(self, negative_slope: float) -> None:
        nn.init.kaiming_uniform_(self.content_projection.weight, a=negative_slope)
        nn.init.zeros_(self.content_projection.bias)

        nn.init.normal_(self.statistic_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.source_embedding.weight, mean=0.0, std=0.02)

    def forward(self, raw_tokens: torch.Tensor) -> torch.Tensor:
        if raw_tokens.ndim != 3 or raw_tokens.shape[1:] != (8, 2):
            raise ValueError(
                "StructuredTokenEncoder expects [B, 8, 2], "
                f"got {tuple(raw_tokens.shape)}"
            )

        content = self.content_projection(raw_tokens)

        statistic_identity = self.statistic_embedding(self.statistic_ids).unsqueeze(0)

        source_identity = self.source_embedding(self.source_ids).unsqueeze(0)

        encoded = content + statistic_identity + source_identity

        return F.leaky_relu(encoded, negative_slope=self.negative_slope)


class StructuredPairedCrossInteraction(nn.Module):
    def __init__(self, token_dim: int, negative_slope: float) -> None:
        super().__init__()

        self.token_dim = token_dim
        self.negative_slope = negative_slope

        self.cross_projection = nn.Linear(2 * token_dim, 2 * token_dim)

        nn.init.kaiming_uniform_(self.cross_projection.weight, a=negative_slope)
        nn.init.zeros_(self.cross_projection.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (8, self.token_dim):
            raise ValueError(
                f"SCR expects [B, 8, {self.token_dim}], " f"got {tuple(tokens.shape)}"
            )

        image_tokens = tokens[:, :4, :]
        edge_tokens = tokens[:, 4:, :]

        product = image_tokens * edge_tokens
        difference = torch.abs(image_tokens - edge_tokens)

        cross_signal = torch.cat([product, difference], dim=-1)

        updates = self.cross_projection(cross_signal)
        updates = F.leaky_relu(updates, negative_slope=self.negative_slope)

        image_update, edge_update = torch.chunk(updates, chunks=2, dim=-1)

        interacted_image = image_tokens + image_update
        interacted_edge = edge_tokens + edge_update

        return torch.cat([interacted_image, interacted_edge], dim=1)


class TransformerSelfAttentionInteraction(nn.Module):
    def __init__(self, token_dim: int, num_heads: int) -> None:
        super().__init__()

        if num_heads <= 0:
            raise ValueError("transformer_num_heads must be positive")
        if token_dim % num_heads != 0:
            raise ValueError(
                "token_dim must be divisible by transformer_num_heads: "
                f"{token_dim} vs {num_heads}"
            )

        self.token_dim = token_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=num_heads, dropout=0.0, batch_first=True
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (8, self.token_dim):
            raise ValueError(
                "Transformer interaction expects "
                f"[B, 8, {self.token_dim}], "
                f"got {tuple(tokens.shape)}"
            )

        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)

        return tokens + attended


class EPCC(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        interaction_type = cfg.interaction_type.lower().strip()

        if interaction_type not in {"none", "scr", "transformer"}:
            raise ValueError(
                "interaction_type must be 'none', 'scr', or 'transformer', "
                f"got {cfg.interaction_type!r}"
            )

        self.interaction_type = interaction_type
        self.token_dim = cfg.token_dim

        self.feature_extractor = EPCCFeatureExtractor(
            cfg.dark_threshold,
            cfg.saturation_threshold,
            cfg.edge_operator,
            use_topk_extreme_statistics=cfg.use_topk_extreme_statistics,
            topk_extreme_k=cfg.topk_extreme_k,
            use_robust_maximum_statistics=cfg.use_robust_maximum_statistics,
            robust_maximum_k=cfg.robust_maximum_k,
        )

        self.token_encoder = StructuredTokenEncoder(
            token_dim=cfg.token_dim, negative_slope=cfg.leaky_relu_slope
        )

        regression_input_dim = cfg.regression_input_dim()

        self.regressor = FastKANRegressor(cfg, input_dim=regression_input_dim)

        if interaction_type == "none":
            self.interaction = nn.Identity()
        elif interaction_type == "scr":
            self.interaction = StructuredPairedCrossInteraction(
                token_dim=cfg.token_dim, negative_slope=cfg.leaky_relu_slope
            )
        else:
            self.interaction = TransformerSelfAttentionInteraction(
                token_dim=cfg.token_dim, num_heads=cfg.transformer_num_heads
            )

    @staticmethod
    def ratios_to_rgb(two_ratios: torch.Tensor) -> torch.Tensor:
        red_over_green = two_ratios[..., 0:1]
        blue_over_green = two_ratios[..., 1:2]
        green = torch.ones_like(red_over_green)

        return torch.cat([red_over_green, green, blue_over_green], dim=-1)

    @staticmethod
    def features_to_tokens(features: torch.Tensor) -> torch.Tensor:
        _validate_16d_features(features)
        return features.reshape(features.shape[0], 8, 2)

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        raw_tokens = self.features_to_tokens(features)

        encoded_tokens = self.token_encoder(raw_tokens)

        interacted_tokens = self.interaction(encoded_tokens)

        fused_features = interacted_tokens.flatten(start_dim=1)

        two_ratios = self.regressor(fused_features)

        return self.ratios_to_rgb(two_ratios)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return self.predict_from_features(features)

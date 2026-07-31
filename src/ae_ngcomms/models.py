"""Fully connected symmetric and asymmetric autoencoders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from torch import Tensor, nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "none": nn.Identity,
}


def _activation(name: str, kwargs: dict[str, Any] | None = None) -> nn.Module:
    try:
        activation_type = ACTIVATIONS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(
            f"Unknown activation {name!r}; choose one of: {choices}"
        ) from exc
    return activation_type(**(kwargs or {}))


def _per_layer(
    value: str | float | Sequence[str] | Sequence[float],
    length: int,
    label: str,
) -> list[str] | list[float]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return [value] * length
    values = list(value)
    if len(values) != length:
        raise ValueError(
            f"{label} has {len(values)} values but the branch has {length} hidden layers"
        )
    return values


def build_mlp_block(
    in_features: int,
    out_features: int,
    *,
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
    layer_norm: bool = False,
    activation_kwargs: dict[str, Any] | None = None,
) -> nn.Sequential:
    """Build ``Linear -> optional norm -> activation -> optional dropout``."""
    if batch_norm and layer_norm:
        raise ValueError("batch_norm and layer_norm are mutually exclusive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the interval [0, 1)")

    layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_features))
    elif layer_norm:
        layers.append(nn.LayerNorm(out_features))
    layers.append(_activation(activation, activation_kwargs))
    if dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Parametrizable multilayer-perceptron encoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] | None = None,
        *,
        activation: str | Sequence[str] = "relu",
        latent_activation: str = "none",
        dropout: float | Sequence[float] = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
        activation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")

        hidden = list(hidden_dims or ())
        if any(width <= 0 for width in hidden):
            raise ValueError("all hidden dimensions must be positive")
        activations = _per_layer(activation, len(hidden), "activation")
        dropouts = _per_layer(dropout, len(hidden), "dropout")

        dims = [input_dim, *hidden]
        blocks = [
            build_mlp_block(
                dims[index],
                dims[index + 1],
                activation=str(activations[index]),
                dropout=float(dropouts[index]),
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                activation_kwargs=activation_kwargs,
            )
            for index in range(len(hidden))
        ]

        self.hidden = nn.Sequential(*blocks)
        self.latent_projection = nn.Linear(dims[-1], latent_dim)
        self.latent_activation = _activation(latent_activation)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden

    def forward(self, inputs: Tensor) -> Tensor:
        return self.latent_activation(self.latent_projection(self.hidden(inputs)))

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, "
            f"latent_dim={self.latent_dim}"
        )


class Decoder(nn.Module):
    """Parametrizable multilayer-perceptron decoder."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] | None = None,
        *,
        activation: str | Sequence[str] = "relu",
        output_activation: str = "none",
        dropout: float | Sequence[float] = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
        activation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or output_dim <= 0:
            raise ValueError("latent_dim and output_dim must be positive")

        hidden = list(hidden_dims or ())
        if any(width <= 0 for width in hidden):
            raise ValueError("all hidden dimensions must be positive")
        activations = _per_layer(activation, len(hidden), "activation")
        dropouts = _per_layer(dropout, len(hidden), "dropout")

        dims = [latent_dim, *hidden]
        blocks = [
            build_mlp_block(
                dims[index],
                dims[index + 1],
                activation=str(activations[index]),
                dropout=float(dropouts[index]),
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                activation_kwargs=activation_kwargs,
            )
            for index in range(len(hidden))
        ]

        self.hidden = nn.Sequential(*blocks)
        self.output_projection = nn.Linear(dims[-1], output_dim)
        self.output_activation = _activation(output_activation)
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden

    def forward(self, latent: Tensor) -> Tensor:
        return self.output_activation(self.output_projection(self.hidden(latent)))

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}"
        )


class AsymmetricAutoencoder(nn.Module):
    """Compose independently configurable encoder and decoder branches."""

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        if encoder.latent_dim != decoder.latent_dim:
            raise ValueError(
                "encoder and decoder latent dimensions differ: "
                f"{encoder.latent_dim} != {decoder.latent_dim}"
            )
        self.encoder = encoder
        self.decoder = decoder

    @property
    def input_dim(self) -> int:
        return self.encoder.input_dim

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def output_dim(self) -> int:
        return self.decoder.output_dim

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(inputs)
        return self.decode(latent), latent


def hidden_widths(start: int, end: int, hidden_layers: int) -> list[int]:
    """Return geometrically spaced hidden widths, matching the study protocol."""
    if start <= 0 or end <= 0:
        raise ValueError("start and end dimensions must be positive")
    if hidden_layers < 0:
        raise ValueError("hidden_layers cannot be negative")
    if hidden_layers == 0:
        return []
    points = np.geomspace(start, end, hidden_layers + 2)
    return [round(float(point)) for point in points[1:-1]]


def branch_depths(depth_index: int, asymmetric: bool) -> tuple[int, int]:
    """Map the paper's depth index to encoder and decoder hidden-layer counts."""
    if depth_index < 0:
        raise ValueError("depth_index cannot be negative")
    if not asymmetric:
        return depth_index, depth_index
    encoder_depth = (depth_index - 1) // 2 if depth_index else 0
    return encoder_depth, depth_index - encoder_depth


def build_autoencoder(
    input_dim: int,
    latent_dim: int,
    depth_index: int,
    *,
    asymmetric: bool,
    hidden_activation: str = "elu",
    latent_activation: str = "selu",
    output_activation: str = "sigmoid",
) -> AsymmetricAutoencoder:
    """Build one architecture from the exact rules used in the article."""
    encoder_depth, decoder_depth = branch_depths(depth_index, asymmetric)
    encoder = Encoder(
        input_dim,
        latent_dim,
        hidden_widths(input_dim, latent_dim, encoder_depth),
        activation=hidden_activation,
        latent_activation=latent_activation,
    )
    decoder = Decoder(
        latent_dim,
        input_dim,
        hidden_widths(latent_dim, input_dim, decoder_depth),
        activation=hidden_activation,
        output_activation=output_activation,
    )
    return AsymmetricAutoencoder(encoder, decoder)


def linear_macs(module: nn.Module) -> int:
    """Count inference multiply-accumulates contributed by linear layers."""
    return sum(
        layer.in_features * layer.out_features
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
    )

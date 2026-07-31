"""Training utilities for the reconstruction autoencoders."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .models import AsymmetricAutoencoder

LOSSES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
}


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a repeatable training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one training run."""

    epochs: int = 300
    learning_rate: float = 2e-3
    loss: str = "mse"
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-7
    batch_size: int = 128
    validation_batch_size: int = 256
    device: str = "cpu"
    log_every: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.loss not in LOSSES:
            raise ValueError(f"unknown loss {self.loss!r}")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    epochs_completed: int = 0


class EarlyStopping:
    """Keep the best model and stop after a configurable validation plateau."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0
        self.best_state: dict[str, Any] | None = None

    def update(self, metric: float, model: nn.Module) -> bool:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.wait = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
        return self.wait >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _inputs(
    batch: Tensor | list[Tensor] | tuple[Tensor, ...], device: torch.device
) -> Tensor:
    tensor = batch[0] if isinstance(batch, (list, tuple)) else batch
    return tensor.to(device)


def _epoch(
    model: AsymmetricAutoencoder,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            inputs = _inputs(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(inputs)
            loss = loss_function(reconstruction, inputs)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach())
            batches += 1
    return total / max(batches, 1)


def fit(
    model: AsymmetricAutoencoder,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainConfig,
) -> TrainHistory:
    """Train a model and restore the lowest-validation-loss checkpoint."""
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = LOSSES[config.loss]()
    stopper = (
        EarlyStopping(
            config.early_stopping_patience,
            config.early_stopping_min_delta,
        )
        if config.early_stopping_patience
        else None
    )
    history = TrainHistory()

    for epoch in range(1, config.epochs + 1):
        train_loss = _epoch(model, train_loader, loss_function, device, optimizer)
        validation_loss = _epoch(
            model,
            validation_loader,
            loss_function,
            device,
            optimizer=None,
        )
        history.train_loss.append(train_loss)
        history.validation_loss.append(validation_loss)
        history.epochs_completed = epoch

        if config.log_every and (epoch == 1 or epoch % config.log_every == 0):
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.7f} "
                f"validation_loss={validation_loss:.7f}"
            )

        if stopper is not None and stopper.update(validation_loss, model):
            break

    if stopper is not None:
        stopper.restore(model)
    return history

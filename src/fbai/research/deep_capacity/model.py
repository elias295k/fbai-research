"""Optional deterministic CPU Torch MLP for the deep-capacity audit."""

from __future__ import annotations

import hashlib
import importlib.util
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from fbai.core.metrics import CLASS_ORDER, validate_probabilities
from fbai.research.deep_capacity.config import DeepCapacityConfig


class DeepCapacityDependencyError(ImportError):
    """Raised when optional Torch support is unavailable."""


class DeepCapacityTrainingError(RuntimeError):
    """Raised when deep-capacity fitting produces an invalid state."""


def deep_capacity_available() -> bool:
    """Return whether the existing optional Torch dependency is installed."""

    return importlib.util.find_spec("torch") is not None


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise DeepCapacityDependencyError(
            "Deep-capacity research requires optional PyTorch support; install "
            "'football-outcome-lab[match2vec]'."
        ) from exc
    return torch


def _set_deterministic_state(seed: int) -> Any:
    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return torch


def _build_network(config: DeepCapacityConfig) -> Any:
    torch = _set_deterministic_state(config.random_seed)
    layers: list[Any] = []
    previous = config.input_feature_count
    for width in config.hidden_dimensions:
        layers.extend(
            [
                torch.nn.Linear(previous, width),
                torch.nn.ReLU(),
                torch.nn.Dropout(config.dropout),
            ]
        )
        previous = width
    layers.append(torch.nn.Linear(previous, config.output_class_count))
    return torch.nn.Sequential(*layers)


def _class_indices(labels: Sequence[str]) -> npt.NDArray[np.int64]:
    class_index = {label: position for position, label in enumerate(CLASS_ORDER)}
    invalid = sorted(set(labels).difference(class_index))
    if invalid:
        raise ValueError(f"deep-capacity labels contain values outside H/D/A: {invalid}")
    return np.asarray([class_index[label] for label in labels], dtype=np.int64)


def _numeric(
    values: npt.ArrayLike,
    *,
    columns: int,
    name: str,
) -> npt.NDArray[np.float32]:
    numeric = np.asarray(values, dtype=np.float32)
    if numeric.ndim != 2 or numeric.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns")
    if not np.isfinite(numeric).all():
        raise DeepCapacityTrainingError(f"{name} contains non-finite values")
    return numeric


class DeepCapacityModel:
    """Source-defined fully connected H/D/A classifier."""

    def __init__(self, config: DeepCapacityConfig) -> None:
        self.config = config
        self._network: Any | None = None
        self._epochs_trained = 0
        self._best_validation_loss: float | None = None

    @property
    def fitted(self) -> bool:
        return self._network is not None

    @property
    def parameter_count(self) -> int:
        return self.config.parameter_count()

    @property
    def epochs_trained(self) -> int:
        return self._epochs_trained

    @property
    def best_validation_loss(self) -> float:
        if self._best_validation_loss is None:
            raise DeepCapacityTrainingError("deep-capacity model has not been fitted")
        return self._best_validation_loss

    def fit(
        self,
        fit_numeric: npt.ArrayLike,
        fit_labels: Sequence[str],
        validation_numeric: npt.ArrayLike,
        validation_labels: Sequence[str],
    ) -> DeepCapacityModel:
        """Fit a fresh Adam optimizer with chronological early stopping."""

        torch = _set_deterministic_state(self.config.random_seed)
        fit_values = _numeric(
            fit_numeric,
            columns=self.config.input_feature_count,
            name="fit numeric features",
        )
        validation_values = _numeric(
            validation_numeric,
            columns=self.config.input_feature_count,
            name="validation numeric features",
        )
        if len(fit_labels) != len(fit_values) or len(validation_labels) != len(validation_values):
            raise ValueError("deep-capacity labels and numeric rows are not aligned")
        fit_targets_array = _class_indices(fit_labels)
        validation_targets_array = _class_indices(validation_labels)
        if frozenset(fit_labels) != frozenset(CLASS_ORDER):
            raise ValueError("deep-capacity fit rows require all H/D/A classes")

        fit_tensor = torch.from_numpy(fit_values)
        validation_tensor = torch.from_numpy(validation_values)
        fit_targets = torch.from_numpy(fit_targets_array)
        validation_targets = torch.from_numpy(validation_targets_array)
        network = _build_network(self.config)
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        rng = np.random.RandomState(self.config.random_seed)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        bad_epochs = 0
        epochs_trained = 0

        for epoch in range(self.config.max_epochs):
            network.train()
            order = rng.permutation(len(fit_values))
            for start in range(0, len(fit_values), self.config.batch_size):
                index = torch.from_numpy(order[start : start + self.config.batch_size])
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    network(fit_tensor[index]),
                    fit_targets[index],
                )
                if not bool(torch.isfinite(loss)):
                    raise DeepCapacityTrainingError(
                        "deep-capacity model produced non-finite training loss"
                    )
                loss.backward()
                optimizer.step()

            if not all(bool(torch.isfinite(parameter).all()) for parameter in network.parameters()):
                raise DeepCapacityTrainingError(
                    "deep-capacity model produced non-finite learned parameters"
                )
            network.eval()
            with torch.no_grad():
                validation_loss = torch.nn.functional.cross_entropy(
                    network(validation_tensor),
                    validation_targets,
                )
            if not bool(torch.isfinite(validation_loss)):
                raise DeepCapacityTrainingError(
                    "deep-capacity model produced non-finite validation loss"
                )
            value = float(validation_loss.item())
            epochs_trained = epoch + 1
            if value < best_loss - self.config.early_stopping_min_delta:
                best_loss = value
                best_state = {
                    name: tensor.detach().clone() for name, tensor in network.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.early_stopping_patience:
                    break

        if best_state is None or not np.isfinite(best_loss):
            raise DeepCapacityTrainingError(
                "deep-capacity model did not produce a finite fitted state"
            )
        network.load_state_dict(best_state)
        network.eval()
        self._network = network
        self._epochs_trained = epochs_trained
        self._best_validation_loss = best_loss
        return self

    def _fitted_network(self) -> Any:
        if self._network is None:
            raise DeepCapacityTrainingError("deep-capacity model must be fitted before prediction")
        return self._network

    def predict_proba(self, numeric: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return finite normalized probabilities in explicit H, D, A order."""

        torch = _require_torch()
        values = _numeric(
            numeric,
            columns=self.config.input_feature_count,
            name="prediction numeric features",
        )
        network = self._fitted_network()
        network.eval()
        with torch.no_grad():
            probabilities = torch.softmax(network(torch.from_numpy(values)), dim=1)
        output = np.asarray(probabilities.cpu().numpy(), dtype=np.float64)
        output /= output.sum(axis=1, keepdims=True)
        return validate_probabilities(output)

    def state_fingerprint(self) -> str:
        """Return a deterministic digest without exposing learned weights."""

        network = self._fitted_network()
        digest = hashlib.sha256()
        for name, tensor in sorted(network.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

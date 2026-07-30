"""Optional deterministic CPU implementation of the source Match2Vec network."""

from __future__ import annotations

import hashlib
import importlib.util
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from fbai.core.metrics import CLASS_ORDER, validate_probabilities
from fbai.research.match2vec.config import Match2VecConfig
from fbai.research.match2vec.corpus import SequenceBatch


class Match2VecDependencyError(ImportError):
    """Raised when the isolated research dependency is not installed."""


class Match2VecTrainingError(RuntimeError):
    """Raised when candidate fitting produces an invalid numerical state."""


def match2vec_available() -> bool:
    """Return whether the optional PyTorch dependency can be imported."""

    return importlib.util.find_spec("torch") is not None


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise Match2VecDependencyError(
            "Match2Vec requires optional PyTorch support; install "
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


def _build_network(config: Match2VecConfig, *, league_count: int, numeric_count: int) -> Any:
    torch = _set_deterministic_state(config.random_seed)
    nn = torch.nn

    class SequenceNetwork(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(config.descriptor_count, config.encoder_dimension),
                nn.ReLU(),
            )
            self.attention = nn.Linear(config.encoder_dimension, 1)
            self.league_embedding = nn.Embedding(
                league_count,
                config.league_embedding_dimension,
            )
            nn.init.normal_(self.league_embedding.weight, std=0.05)
            head_input = config.representation_feature_count + numeric_count
            self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(head_input, 3))

        def team_state(self, sequence: Any, mask: Any) -> Any:
            encoded = self.encoder(sequence)
            scores = self.attention(encoded).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            weights = weights * mask.any(dim=1, keepdim=True)
            return torch.einsum("bk,bke->be", weights, encoded)

        def representation(
            self,
            home_sequence: Any,
            home_mask: Any,
            away_sequence: Any,
            away_mask: Any,
            league_id: Any,
        ) -> Any:
            home_state = self.team_state(home_sequence, home_mask)
            away_state = self.team_state(away_sequence, away_mask)
            return torch.cat(
                [
                    home_state,
                    away_state,
                    home_state - away_state,
                    self.league_embedding(league_id),
                ],
                dim=1,
            )

        def forward(
            self,
            home_sequence: Any,
            home_mask: Any,
            away_sequence: Any,
            away_mask: Any,
            league_id: Any,
            numeric: Any,
        ) -> Any:
            representation = self.representation(
                home_sequence,
                home_mask,
                away_sequence,
                away_mask,
                league_id,
            )
            return self.head(torch.cat([representation, numeric], dim=1))

    return SequenceNetwork()


def _class_indices(labels: Sequence[str]) -> npt.NDArray[np.int64]:
    class_index = {label: position for position, label in enumerate(CLASS_ORDER)}
    invalid = sorted(set(labels).difference(class_index))
    if invalid:
        raise ValueError(f"Match2Vec labels contain values outside H/D/A: {invalid}")
    return np.asarray([class_index[label] for label in labels], dtype=np.int64)


def _validate_numeric(
    values: npt.ArrayLike,
    *,
    rows: int,
    columns: int,
    name: str,
) -> npt.NDArray[np.float32]:
    numeric = np.asarray(values, dtype=np.float32)
    if numeric.shape != (rows, columns):
        raise ValueError(f"{name} must have shape {(rows, columns)}, got {numeric.shape}")
    if not np.isfinite(numeric).all():
        raise Match2VecTrainingError(f"{name} contains non-finite values")
    return numeric


class Match2VecSequenceModel:
    """Attention-pooled sequence representation plus the source linear H/D/A head."""

    def __init__(
        self,
        *,
        league_count: int,
        numeric_feature_count: int,
        config: Match2VecConfig | None = None,
    ) -> None:
        if league_count < 1 or numeric_feature_count < 1:
            raise ValueError("league_count and numeric_feature_count must be positive")
        self.config = config or Match2VecConfig()
        self.league_count = league_count
        self.numeric_feature_count = numeric_feature_count
        self._network: Any | None = None
        self._epochs_trained = 0
        self._best_validation_loss: float | None = None

    @property
    def fitted(self) -> bool:
        return self._network is not None

    @property
    def epochs_trained(self) -> int:
        return self._epochs_trained

    @property
    def best_validation_loss(self) -> float:
        if self._best_validation_loss is None:
            raise Match2VecTrainingError("Match2Vec has not been fitted")
        return self._best_validation_loss

    @property
    def representation_feature_count(self) -> int:
        return self.config.representation_feature_count

    def _inputs(
        self,
        batch: SequenceBatch,
        league_ids: npt.ArrayLike,
        numeric: npt.ArrayLike,
    ) -> dict[str, Any]:
        torch = _require_torch()
        leagues = np.asarray(league_ids, dtype=np.int64)
        if leagues.shape != (batch.row_count,):
            raise ValueError("league_ids must contain one ID per sequence row")
        if (leagues < 0).any() or (leagues >= self.league_count).any():
            raise ValueError("league_ids contain an ID outside the fitted vocabulary")
        numeric_values = _validate_numeric(
            numeric,
            rows=batch.row_count,
            columns=self.numeric_feature_count,
            name="numeric features",
        )
        return {
            "home_sequence": torch.from_numpy(np.asarray(batch.home_sequences).copy()),
            "home_mask": torch.from_numpy(np.asarray(batch.home_mask).copy()),
            "away_sequence": torch.from_numpy(np.asarray(batch.away_sequences).copy()),
            "away_mask": torch.from_numpy(np.asarray(batch.away_mask).copy()),
            "league_id": torch.from_numpy(leagues),
            "numeric": torch.from_numpy(numeric_values),
        }

    def fit(
        self,
        *,
        fit_batch: SequenceBatch,
        fit_league_ids: npt.ArrayLike,
        fit_numeric: npt.ArrayLike,
        fit_labels: Sequence[str],
        validation_batch: SequenceBatch,
        validation_league_ids: npt.ArrayLike,
        validation_numeric: npt.ArrayLike,
        validation_labels: Sequence[str],
    ) -> Match2VecSequenceModel:
        """Fit from scratch with Adam and source-verified chronological early stopping."""

        if len(fit_labels) != fit_batch.row_count:
            raise ValueError("fit labels and sequence rows are not aligned")
        if len(validation_labels) != validation_batch.row_count:
            raise ValueError("validation labels and sequence rows are not aligned")
        torch = _set_deterministic_state(self.config.random_seed)
        fit_inputs = self._inputs(fit_batch, fit_league_ids, fit_numeric)
        validation_inputs = self._inputs(
            validation_batch,
            validation_league_ids,
            validation_numeric,
        )
        fit_targets = torch.from_numpy(_class_indices(fit_labels))
        validation_targets = torch.from_numpy(_class_indices(validation_labels))
        network = _build_network(
            self.config,
            league_count=self.league_count,
            numeric_count=self.numeric_feature_count,
        )
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
            order = rng.permutation(fit_batch.row_count)
            for start in range(0, fit_batch.row_count, self.config.batch_size):
                index = torch.from_numpy(order[start : start + self.config.batch_size])
                current_inputs = {name: values[index] for name, values in fit_inputs.items()}
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    network(**current_inputs),
                    fit_targets[index],
                )
                if not bool(torch.isfinite(loss)):
                    raise Match2VecTrainingError("Match2Vec produced a non-finite training loss")
                loss.backward()
                optimizer.step()

            if not all(bool(torch.isfinite(parameter).all()) for parameter in network.parameters()):
                raise Match2VecTrainingError("Match2Vec produced non-finite learned parameters")
            network.eval()
            with torch.no_grad():
                validation_loss = torch.nn.functional.cross_entropy(
                    network(**validation_inputs),
                    validation_targets,
                )
            if not bool(torch.isfinite(validation_loss)):
                raise Match2VecTrainingError("Match2Vec produced a non-finite validation loss")
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
            raise Match2VecTrainingError("Match2Vec did not produce a finite fitted state")
        network.load_state_dict(best_state)
        network.eval()
        self._network = network
        self._epochs_trained = epochs_trained
        self._best_validation_loss = best_loss
        return self

    def _fitted_network(self) -> Any:
        if self._network is None:
            raise Match2VecTrainingError("Match2Vec must be fitted before prediction")
        return self._network

    def predict_proba(
        self,
        batch: SequenceBatch,
        league_ids: npt.ArrayLike,
        numeric: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Return normalized probabilities in the explicit H, D, A order."""

        torch = _require_torch()
        network = self._fitted_network()
        inputs = self._inputs(batch, league_ids, numeric)
        network.eval()
        with torch.no_grad():
            probabilities = torch.softmax(network(**inputs), dim=1).cpu().numpy()
        values = np.asarray(probabilities, dtype=np.float64)
        values /= values.sum(axis=1, keepdims=True)
        return validate_probabilities(values)

    def transform(
        self,
        batch: SequenceBatch,
        league_ids: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Expose the learned 100-value source match representation."""

        torch = _require_torch()
        network = self._fitted_network()
        leagues = np.asarray(league_ids, dtype=np.int64)
        if leagues.shape != (batch.row_count,):
            raise ValueError("league_ids must contain one ID per sequence row")
        network.eval()
        with torch.no_grad():
            values = (
                network.representation(
                    torch.from_numpy(np.asarray(batch.home_sequences).copy()),
                    torch.from_numpy(np.asarray(batch.home_mask).copy()),
                    torch.from_numpy(np.asarray(batch.away_sequences).copy()),
                    torch.from_numpy(np.asarray(batch.away_mask).copy()),
                    torch.from_numpy(leagues),
                )
                .cpu()
                .numpy()
            )
        result = np.asarray(values, dtype=np.float64)
        expected = (batch.row_count, self.representation_feature_count)
        if result.shape != expected:
            raise Match2VecTrainingError(
                f"Match2Vec representation has shape {result.shape}, expected {expected}"
            )
        if not np.isfinite(result).all():
            raise Match2VecTrainingError("Match2Vec produced non-finite representation features")
        return result

    def state_fingerprint(self) -> str:
        """Return a deterministic digest for tests without exposing model parameters."""

        network = self._fitted_network()
        digest = hashlib.sha256()
        for name, tensor in sorted(network.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

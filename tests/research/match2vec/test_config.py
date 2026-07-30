from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from fbai.research.match2vec import model as model_module
from fbai.research.match2vec.config import Match2VecConfig
from fbai.research.match2vec.model import Match2VecDependencyError


def test_source_verified_defaults_are_exact() -> None:
    config = Match2VecConfig()

    assert config.variant == "M2V-SEQ"
    assert config.sequence_length == 10
    assert config.descriptor_count == 13
    assert config.encoder_dimension == 32
    assert config.league_embedding_dimension == 4
    assert config.use_lr52_numeric
    assert config.optimizer == "Adam"
    assert config.learning_rate == 3e-3
    assert config.weight_decay == 1e-4
    assert config.batch_size == 512
    assert config.max_epochs == 300
    assert config.early_stopping_patience == 15
    assert config.random_seed == 42
    assert config.device == "cpu"
    assert config.negative_sample_count == 0
    assert config.representation_feature_count == 100


def test_configuration_is_immutable_and_json_safe() -> None:
    config = Match2VecConfig()

    with pytest.raises(FrozenInstanceError):
        config.sequence_length = 5  # type: ignore[misc]
    assert json.loads(json.dumps(config.to_dict())) == config.to_dict()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence_length": 0}, "sequence_length"),
        ({"encoder_dimension": 0}, "dimensions"),
        ({"max_epochs": 0}, "positive"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"device": "cuda"}, "CPU"),
        ({"negative_sample_count": 1}, "negative sampling"),
    ],
)
def test_invalid_configuration_fails(
    kwargs: dict[str, str | int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Match2VecConfig(**kwargs)  # type: ignore[arg-type]


def test_importing_core_package_does_not_import_torch() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import fbai; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_missing_optional_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def import_without_torch(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    with pytest.raises(Match2VecDependencyError, match=r"\[match2vec\]"):
        model_module._require_torch()

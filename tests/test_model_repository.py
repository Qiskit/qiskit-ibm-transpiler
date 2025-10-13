# -*- coding: utf-8 -*-

"""Unit tests for the RL synthesis repository."""

from unittest.mock import MagicMock

import pytest

from qiskit_gym.rl import RLSynthesis

from qiskit_ibm_transpiler.model_repository import RLSynthesisRepository


def test_register_and_get_returns_same_instance():
    repo = RLSynthesisRepository()
    model = MagicMock(spec=RLSynthesis)

    repo.register("qiskit", "topology-abc", model)

    assert repo.get("qiskit", "topology-abc") is model


def test_register_from_config_uses_rlsynthesis_classmethod(monkeypatch):
    repo = RLSynthesisRepository()
    model = MagicMock(spec=RLSynthesis)

    def _fake_from_config(cls, config_path, model_path=None):
        assert config_path == "config.json"
        assert model_path == "model.pt"
        return model

    monkeypatch.setattr(
        RLSynthesis,
        "from_config_json",
        classmethod(_fake_from_config),
    )

    returned = repo.register_from_config(
        "az", "hash", "config.json", model_path="model.pt"
    )

    assert returned is model
    assert repo.get("az", "hash") is model


def test_get_raises_key_error_when_missing():
    repo = RLSynthesisRepository()

    with pytest.raises(KeyError):
        repo.get("az", "missing")

# -*- coding: utf-8 -*-

"""Unit tests for model bootstrap helpers."""

import json
from unittest.mock import MagicMock

import pytest

from qiskit_ibm_transpiler import model_bootstrap


def test_ensure_models_loaded_registers_permutation_models(tmp_path, monkeypatch):
    monkeypatch.setenv("QISKIT_TRANSPILER_PERMUTATION_REPO_ID", "ibm/permutation")
    monkeypatch.setenv("QISKIT_TRANSPILER_PERMUTATION_REVISION", "main")

    model_bootstrap.reset_model_repository()

    config_dir = tmp_path / "permutation"
    config_dir.mkdir()
    config_path = config_dir / "model.json"
    config_path.write_text(
        json.dumps(
            {
                "env": {
                    "gateset": [["SWAP", [0, 1]], ["SWAP", [1, 0]]],
                }
            }
        ),
        encoding="utf-8",
    )

    calls = {"count": 0}

    class _DummyHF:
        def __init__(self, *_, **__):
            pass

        def download_models(self, repo_id, revision):
            calls["count"] += 1
            assert repo_id == "ibm/permutation"
            assert revision == "main"
            return str(tmp_path)

    dummy_model = MagicMock(name="rl_model")

    monkeypatch.setattr(model_bootstrap, "HFInterface", _DummyHF)

    monkeypatch.setitem(
        model_bootstrap.TYPE_CONFIGS,
        "permutation",
        model_bootstrap.ModelTypeConfig(
            repo_env="QISKIT_TRANSPILER_PERMUTATION_REPO_ID",
            revision_env="QISKIT_TRANSPILER_PERMUTATION_REVISION",
            subdir_env="QISKIT_TRANSPILER_PERMUTATION_SUBDIR",
            hash_fn=lambda _: "hash123",
            loader=lambda config_path, model_path=None: dummy_model,
        ),
    )

    repo = model_bootstrap.ensure_models_loaded("permutation")

    record = repo.get("hash123")
    assert record.model is dummy_model
    assert record.coupling_map == [(0, 1)]
    assert calls["count"] == 1

    assert model_bootstrap.ensure_models_loaded("permutation") is repo
    assert calls["count"] == 1

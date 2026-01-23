# -*- coding: utf-8 -*-

"""Unit tests for the Hugging Face integration helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from qiskit_ibm_transpiler.hf_models_client import HFInterface, _is_version


def _mock_hf_api(monkeypatch):
    """Return a mock and route :class:`~huggingface_hub.HfApi` to it."""

    mock_api = MagicMock()
    monkeypatch.setattr(
        "qiskit_ibm_transpiler.hf_models_client.HfApi", lambda **_: mock_api
    )
    return mock_api


def test_is_version_true():
    assert _is_version("1.2.3") is True


def test_is_version_false():
    assert _is_version("latest") is False


def test_get_rev_returns_literal_when_not_spec(monkeypatch):
    mock_api = _mock_hf_api(monkeypatch)
    interface = HFInterface()

    literal = interface._get_rev_(repo_id="ibm/test", revision="main")

    assert literal == "main"
    mock_api.list_repo_refs.assert_not_called()


def test_get_rev_resolves_specifier(monkeypatch):
    mock_api = _mock_hf_api(monkeypatch)
    # ``list_repo_refs`` returns objects with a ``name`` attribute; ``SimpleNamespace``
    # lets us model that without creating a custom class.
    mock_api.list_repo_refs.return_value = SimpleNamespace(
        tags=[
            SimpleNamespace(name="0.9.0"),
            SimpleNamespace(name="1.1.0"),
            SimpleNamespace(name="1.3.5"),
        ]
    )

    interface = HFInterface()
    resolved = interface._get_rev_(repo_id="ibm/test", revision=">=1.0,<2.0")

    assert resolved == "1.3.5"
    mock_api.list_repo_refs.assert_called_once_with(repo_id="ibm/test")


def test_get_rev_raises_when_no_candidate(monkeypatch):
    mock_api = _mock_hf_api(monkeypatch)
    mock_api.list_repo_refs.return_value = SimpleNamespace(
        tags=[SimpleNamespace(name="0.5.0")]
    )

    interface = HFInterface()

    with pytest.raises(RuntimeError, match="Revision "):
        interface._get_rev_(repo_id="ibm/test", revision=">=1.0")


def test_download_models_uses_resolved_revision(monkeypatch):
    mock_api = _mock_hf_api(monkeypatch)
    mock_api.snapshot_download.return_value = "/tmp/models"
    mock_api.list_repo_refs.return_value = SimpleNamespace(
        tags=[SimpleNamespace(name="1.3.5")]
    )

    interface = HFInterface()

    path = interface.download_models(repo_id="ibm/test", revision=">=1.0")

    assert path == "/tmp/models"
    mock_api.snapshot_download.assert_called_once_with(
        repo_id="ibm/test", revision="1.3.5"
    )

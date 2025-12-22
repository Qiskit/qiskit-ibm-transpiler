# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit tests for _get_credentials_from_system function"""

import json
from pathlib import Path

import pytest

from qiskit_ibm_transpiler.wrappers.base import _get_credentials_from_system


@pytest.fixture
def temp_qiskit_dir(tmp_path, monkeypatch):
    """Create a temporary .qiskit directory for testing."""
    qiskit_dir = tmp_path / ".qiskit"
    qiskit_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return qiskit_dir


@pytest.fixture
def clear_env_vars(monkeypatch):
    """Ensure both QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE env vars are not set."""
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    monkeypatch.delenv("QISKIT_IBM_INSTANCE", raising=False)


class TestGetCredentialsFromSystem:
    """Tests for the _get_credentials_from_system function."""

    def test_credentials_from_env_vars(self, monkeypatch):
        """Test that credentials are retrieved from env vars."""
        expected_token = "env_token_12345"
        expected_instance = "crn:123:instance"
        monkeypatch.setenv("QISKIT_IBM_TOKEN", expected_token)
        monkeypatch.setenv("QISKIT_IBM_INSTANCE", expected_instance)

        credentials = _get_credentials_from_system()

        assert credentials["token"] == expected_token
        assert credentials["instance"] == expected_instance

    def test_credentials_from_env_token_only(self, monkeypatch):
        """Test that credentials work with only token in env var."""
        expected_token = "env_token_12345"
        monkeypatch.setenv("QISKIT_IBM_TOKEN", expected_token)
        monkeypatch.delenv("QISKIT_IBM_INSTANCE", raising=False)

        credentials = _get_credentials_from_system()

        assert credentials["token"] == expected_token
        assert credentials["instance"] is None

    def test_credentials_from_file_with_instance(self, temp_qiskit_dir, clear_env_vars):
        """Test that both token and instance are retrieved from file."""
        expected_token = "file_token_12345"
        expected_instance = "crn:456:instance"
        creds = {
            "default-ibm-quantum-platform": {
                "token": expected_token,
                "instance": expected_instance,
            }
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))

        credentials = _get_credentials_from_system()

        assert credentials["token"] == expected_token
        assert credentials["instance"] == expected_instance

    def test_credentials_from_file_without_instance(
        self, temp_qiskit_dir, clear_env_vars
    ):
        """Test that credentials work when instance is not in file."""
        expected_token = "file_token_12345"
        creds = {
            "default-ibm-quantum-platform": {
                "token": expected_token,
            }
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))

        credentials = _get_credentials_from_system()

        assert credentials["token"] == expected_token
        assert credentials["instance"] is None

    def test_credentials_fallback_to_default_ibm_quantum(
        self, temp_qiskit_dir, clear_env_vars
    ):
        """Test fallback to default-ibm-quantum when platform key is missing."""
        expected_token = "legacy_token_12345"
        expected_instance = "legacy_instance"
        creds = {
            "default-ibm-quantum": {
                "token": expected_token,
                "instance": expected_instance,
            },
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))

        credentials = _get_credentials_from_system()

        assert credentials["token"] == expected_token
        assert credentials["instance"] == expected_instance

    def test_credentials_from_specified_account_name(
        self, temp_qiskit_dir, clear_env_vars
    ):
        """Test credentials are retrieved from a specified account_name."""
        expected_token = "custom_token_12345"
        expected_instance = "crn:789:custom"
        creds = {
            "my-custom-account": {
                "token": expected_token,
                "instance": expected_instance,
            },
            "default-ibm-quantum-platform": {
                "token": "default_token",
                "instance": "default_instance",
            },
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))

        credentials = _get_credentials_from_system(account_name="my-custom-account")

        assert credentials["token"] == expected_token
        assert credentials["instance"] == expected_instance

    def test_credentials_fallback_to_default_when_account_not_found(
        self, temp_qiskit_dir, clear_env_vars
    ):
        """Test fallback to default accounts when specified account_name is not found."""
        expected_token = "default_platform_token"
        expected_instance = "default_instance"
        creds = {
            "default-ibm-quantum-platform": {
                "token": expected_token,
                "instance": expected_instance,
            },
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))

        credentials = _get_credentials_from_system(account_name="non-existent-account")

        assert credentials["token"] == expected_token
        assert credentials["instance"] == expected_instance

    def test_no_credentials_file_raises_exception(
        self, temp_qiskit_dir, clear_env_vars
    ):
        """Test that an exception is raised when credentials file doesn't exist."""
        with pytest.raises(Exception) as exc_info:
            _get_credentials_from_system()

        assert "does not exist" in str(exc_info.value)

    def test_no_token_in_any_account_raises_exception(
        self, temp_qiskit_dir, clear_env_vars
    ):
        """Test that an exception is raised when no account has a token."""
        creds = {
            "some-account": {"url": "https://example.com"},  # No token
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))

        with pytest.raises(Exception) as exc_info:
            _get_credentials_from_system()

        assert "No valid account with token found" in str(exc_info.value)

    def test_empty_credentials_file_raises_exception(
        self, temp_qiskit_dir, clear_env_vars
    ):
        """Test that an exception is raised when credentials file is empty."""
        creds = {}
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))

        with pytest.raises(Exception) as exc_info:
            _get_credentials_from_system()

        assert "No valid account with token found" in str(exc_info.value)

    def test_no_token_with_account_name_raises_exception(
        self, temp_qiskit_dir, clear_env_vars
    ):
        """Test that an exception is raised when neither account_name nor defaults have a token."""
        creds = {
            "some-account": {"url": "https://example.com"},  # No token
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))

        with pytest.raises(Exception) as exc_info:
            _get_credentials_from_system(account_name="my-custom-account")

        assert "my-custom-account" in str(exc_info.value)
        assert "No valid account with token found" in str(exc_info.value)

    def test_env_vars_take_priority_over_file(self, temp_qiskit_dir, monkeypatch):
        """Test that env vars take priority over file credentials."""
        env_token = "env_priority_token"
        env_instance = "env_priority_instance"
        file_token = "file_token"
        file_instance = "file_instance"

        monkeypatch.setenv("QISKIT_IBM_TOKEN", env_token)
        monkeypatch.setenv("QISKIT_IBM_INSTANCE", env_instance)
        creds = {
            "default-ibm-quantum-platform": {
                "token": file_token,
                "instance": file_instance,
            }
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))

        credentials = _get_credentials_from_system()

        assert credentials["token"] == env_token
        assert credentials["instance"] == env_instance

    def test_default_ibm_quantum_platform_takes_priority(
        self, temp_qiskit_dir, clear_env_vars
    ):
        """Test that default-ibm-quantum-platform account is preferred over default-ibm-quantum."""
        expected_token = "platform_token_12345"
        expected_instance = "platform_instance"
        creds = {
            "default-ibm-quantum-platform": {
                "token": expected_token,
                "instance": expected_instance,
            },
            "default-ibm-quantum": {"token": "legacy_token", "instance": "legacy_inst"},
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))

        credentials = _get_credentials_from_system()

        assert credentials["token"] == expected_token
        assert credentials["instance"] == expected_instance

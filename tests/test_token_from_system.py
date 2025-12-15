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

"""Unit tests for _get_token_from_system function"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from qiskit_ibm_transpiler.wrappers.base import _get_token_from_system


@pytest.fixture
def temp_qiskit_dir(tmp_path, monkeypatch):
    """Create a temporary .qiskit directory for testing."""
    qiskit_dir = tmp_path / ".qiskit"
    qiskit_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return qiskit_dir


@pytest.fixture
def clear_env_token(monkeypatch):
    """Ensure QISKIT_IBM_TOKEN env var is not set during tests."""
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)


class TestGetTokenFromSystem:
    """Tests for the _get_token_from_system function."""

    def test_token_from_env_var(self, monkeypatch):
        """Test that token is retrieved from QISKIT_IBM_TOKEN env var."""
        expected_token = "env_token_12345"
        monkeypatch.setenv("QISKIT_IBM_TOKEN", expected_token)
        
        token = _get_token_from_system()
        
        assert token == expected_token

    def test_token_from_default_ibm_quantum_platform(self, temp_qiskit_dir, clear_env_token):
        """Test that default-ibm-quantum-platform account is preferred."""
        expected_token = "platform_token_12345"
        creds = {
            "default-ibm-quantum-platform": {"token": expected_token},
            "default-ibm-quantum": {"token": "legacy_token"},
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))
        
        token = _get_token_from_system()
        
        assert token == expected_token

    def test_token_fallback_to_default_ibm_quantum(self, temp_qiskit_dir, clear_env_token):
        """Test fallback to default-ibm-quantum when platform key is missing."""
        expected_token = "legacy_token_12345"
        creds = {
            "default-ibm-quantum": {"token": expected_token},
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))
        
        token = _get_token_from_system()
        
        assert token == expected_token

    def test_token_fallback_to_any_account(self, temp_qiskit_dir, clear_env_token):
        """Test fallback to any account with a token when defaults are missing."""
        expected_token = "custom_account_token_12345"
        creds = {
            "my-custom-account": {"token": expected_token},
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))
        
        token = _get_token_from_system()
        
        assert token == expected_token

    def test_no_credentials_file_raises_exception(self, temp_qiskit_dir, clear_env_token):
        """Test that an exception is raised when credentials file doesn't exist."""
        # Don't create the file
        
        with pytest.raises(Exception) as exc_info:
            _get_token_from_system()
        
        assert "does not exist" in str(exc_info.value)

    def test_no_token_in_any_account_raises_exception(self, temp_qiskit_dir, clear_env_token):
        """Test that an exception is raised when no account has a token."""
        creds = {
            "some-account": {"url": "https://example.com"},  # No token
        }
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))
        
        with pytest.raises(Exception) as exc_info:
            _get_token_from_system()
        
        assert "No valid account with token found" in str(exc_info.value)

    def test_empty_credentials_file_raises_exception(self, temp_qiskit_dir, clear_env_token):
        """Test that an exception is raised when credentials file is empty."""
        creds = {}
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))
        
        with pytest.raises(Exception) as exc_info:
            _get_token_from_system()
        
        assert "No valid account with token found" in str(exc_info.value)

    def test_env_token_takes_priority_over_file(self, temp_qiskit_dir, monkeypatch):
        """Test that env var token takes priority over file."""
        env_token = "env_priority_token"
        file_token = "file_token"
        
        monkeypatch.setenv("QISKIT_IBM_TOKEN", env_token)
        creds = {"default-ibm-quantum-platform": {"token": file_token}}
        creds_file = temp_qiskit_dir / "qiskit-ibm.json"
        creds_file.write_text(json.dumps(creds))
        
        token = _get_token_from_system()
        
        assert token == env_token

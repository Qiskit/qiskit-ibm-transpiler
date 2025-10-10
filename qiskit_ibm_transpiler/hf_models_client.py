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

import logging
import os

from packaging.version import Version
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from pathlib import Path

from huggingface_hub import HfApi

def _is_version(t):
    """Return ``True`` when ``t`` can be parsed into a :class:`packaging.version.Version`."""
    try:
        Version(t)
        return True
    except:
        return False
    
class HFInterface:
    """Lightweight wrapper around :mod:`huggingface_hub` for model retrieval."""
    
    hf_api = None
    
    
    def __init__(self, endpoint: str= None, token: str=None):
        """Build a reusable :class:`huggingface_hub.HfApi` client.

        Parameters
        ----------
        endpoint:
            Optional Hugging Face Hub endpoint. Falls back to the
            ``QISKIT_TRANSPILER_HF_ENDPOINT`` environment variable when omitted,
            and ultimately defaults to the public Hugging Face endpoint.
        token:
            Optional access token. Falls back to ``QISKIT_TRANSPILER_HF_TOKEN``
            when not provided; if neither is set, the client accesses only
            public repositories.
        """
        if HFInterface.hf_api == None:
            hf_kwargs = {
                "endpoint": endpoint or os.getenv("QISKIT_TRANSPILER_HF_ENDPOINT"),
                "token": token or os.getenv("QISKIT_TRANSPILER_HF_TOKEN"),
            }
            print(hf_kwargs)
            HFInterface.hf_api = HfApi(**hf_kwargs)

    def _get_rev_(self, repo_id: str, revision: str):
        """Resolve revision specifiers (e.g. ``">=1.2"``) to the latest matching tag."""
        try:
            rev_spec = SpecifierSet(revision)
        except InvalidSpecifier:
            return revision
        refs = self.hf_api.list_repo_refs(repo_id=repo_id)
        tags = [Version(t.name) for t in refs.tags if _is_version(t.name)]
        candidates = [t for t in tags if t in rev_spec]
        if not candidates:
            raise RuntimeError(f"Revision {revision} not found!")
        return str(max(candidates))
    
    def download_models(self, repo_id: str, revision: str) -> Path:
        """Download a model snapshot for ``repo_id`` at ``revision`` to a local cache."""
        revision = self._get_rev_(repo_id=repo_id, revision=revision)
        logging.info(f"Downloading models in {repo_id} for revision {revision}")
        return self.hf_api.snapshot_download(repo_id=repo_id, revision=revision)
        


    

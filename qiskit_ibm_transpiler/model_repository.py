# -*- coding: utf-8 -*-

"""Repository utilities for RL synthesis models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from qiskit_gym.rl import RLSynthesis

from .utils import extract_coupling_edges


@dataclass
class RLSynthesisRecord:
    model: RLSynthesis
    coupling_map: List[Tuple[int, int]]


@dataclass
class RLSynthesisRepository:
    """In-memory index of :class:`qiskit_gym.rl.synthesis.RLSynthesis` models."""

    _store: Dict[str, RLSynthesisRecord] = field(default_factory=dict)

    def register(
        self,
        topology_hash: str,
        model: RLSynthesis,
        coupling_map: List[Tuple[int, int]],
    ) -> None:
        """Insert or replace the model entry for ``topology_hash``."""

        self._store[topology_hash] = RLSynthesisRecord(
            model=model, coupling_map=list(coupling_map)
        )

    def register_from_config(
        self,
        topology_hash: str,
        config_path: str,
        model_path: Optional[str] = None,
    ) -> RLSynthesis:
        """Load and register a model using :meth:`RLSynthesis.from_config_json`."""

        model = RLSynthesis.from_config_json(
            config_path=config_path, model_path=model_path
        )
        coupling_map = extract_coupling_edges(model.env_config)
        self.register(topology_hash, model, coupling_map)
        return model

    def get(self, topology_hash: str) -> RLSynthesisRecord:
        """Return the stored entry for ``topology_hash``."""

        return self._store[topology_hash]

    def __contains__(self, topology_hash: str) -> bool:  # pragma: no cover - delegation
        return topology_hash in self._store

    def __len__(self) -> int:  # pragma: no cover - delegation
        return len(self._store)

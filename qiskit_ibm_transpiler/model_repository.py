# -*- coding: utf-8 -*-

"""Repository utilities for RL synthesis models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from qiskit_gym.rl import RLSynthesis


@dataclass
class RLSynthesisRepository:
    """In-memory index of :class:`qiskit_gym.rl.synthesis.RLSynthesis` models."""

    _store: Dict[tuple[str, str], RLSynthesis] = field(default_factory=dict)

    def register(self, synthesis_type: str, topology_hash: str, model: RLSynthesis) -> None:
        """Insert or replace the model for the ``(synthesis_type, topology_hash)`` key."""

        self._store[(synthesis_type, topology_hash)] = model

    def register_from_config(
        self,
        synthesis_type: str,
        topology_hash: str,
        config_path: str,
        model_path: str | None = None,
    ) -> RLSynthesis:
        """Load and register a model using :meth:`RLSynthesis.from_config_json`."""

        model = RLSynthesis.from_config_json(config_path=config_path, model_path=model_path)
        self.register(synthesis_type, topology_hash, model)
        return model

    def get(self, synthesis_type: str, topology_hash: str) -> RLSynthesis:
        """Return the model for ``(synthesis_type, topology_hash)``."""

        return self._store[(synthesis_type, topology_hash)]

    def __contains__(self, key: tuple[str, str]) -> bool:  # pragma: no cover - delegation
        return key in self._store

    def __len__(self) -> int:  # pragma: no cover - delegation
        return len(self._store)

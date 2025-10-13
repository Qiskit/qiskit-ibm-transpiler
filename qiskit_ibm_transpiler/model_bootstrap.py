# -*- coding: utf-8 -*-

"""Helpers for downloading and caching AI model artefacts."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from threading import RLock
from typing import Callable, Dict, Iterable, Mapping, Optional, Set

from qiskit_gym.rl.synthesis import RLSynthesis

from .hf_models_client import HFInterface
from .model_repository import RLSynthesisRepository
from .utils import compute_topology_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelTypeConfig:
    """Configuration describing how to bootstrap a model family."""

    repo_env: str
    revision_env: str
    subdir_env: Optional[str] = None
    config_glob: str = "*.json"
    default_repo_id: Optional[str] = None
    default_revision: str = "main"
    default_subdir: Optional[str] = None
    env_extractor: Callable[[Mapping[str, object]], Mapping[str, object]] = (
        lambda data: data.get("env", {})  # type: ignore[arg-type]
    )
    hash_fn: Callable[[Mapping[str, object]], str] = compute_topology_hash
    loader: Callable[[Path, Optional[Path]], RLSynthesis] = (
        lambda config_path, model_path=None: RLSynthesis.from_config_json(  # type: ignore[misc]
            str(config_path), model_path=str(model_path) if model_path else None
        )
    )


def _load_static_sources() -> Dict[str, Dict[str, Optional[str]]]:
    try:
        content = resources.read_text(
            "qiskit_ibm_transpiler.data", "model_sources.json", encoding="utf-8"
        )
        return json.loads(content)
    except FileNotFoundError:
        logger.debug("model_sources.json not found; falling back to env-only configuration")
    except ModuleNotFoundError:
        logger.debug("Data package not available; falling back to env-only configuration")
    except Exception as exc:  # pragma: no cover - unexpected
        logger.warning("Failed to load static model sources: %s", exc)
    return {}


_STATIC_SOURCES = _load_static_sources()


TYPE_CONFIGS: Dict[str, ModelTypeConfig] = {
    "permutation": ModelTypeConfig(
        repo_env="QISKIT_TRANSPILER_PERMUTATION_REPO_ID",
        revision_env="QISKIT_TRANSPILER_PERMUTATION_REVISION",
        subdir_env="QISKIT_TRANSPILER_PERMUTATION_SUBDIR",
        default_repo_id=_STATIC_SOURCES.get("permutation", {}).get("repo_id"),
        default_revision=_STATIC_SOURCES.get("permutation", {}).get("revision", "main"),
        default_subdir=_STATIC_SOURCES.get("permutation", {}).get("subdir"),
    ),
}


_BOOTSTRAP_LOCK = RLock()
_BOOTSTRAPPED_TYPES: Set[str] = set()
_REPOSITORIES: Dict[str, RLSynthesisRepository] = {}


def get_model_repository(model_type: str) -> RLSynthesisRepository:
    """Return the repository for ``model_type``, creating it if needed."""

    with _BOOTSTRAP_LOCK:
        repo = _REPOSITORIES.get(model_type)
        if repo is None:
            repo = RLSynthesisRepository()
            _REPOSITORIES[model_type] = repo
        return repo


def reset_model_repository() -> None:
    """Reset bootstrap state (testing helper)."""

    with _BOOTSTRAP_LOCK:
        for repo in _REPOSITORIES.values():
            repo._store.clear()  # type: ignore[attr-defined]
        _REPOSITORIES.clear()
        _BOOTSTRAPPED_TYPES.clear()


def ensure_models_loaded(
    model_type: str,
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    subdir: Optional[str] = None,
) -> RLSynthesisRepository:
    """Ensure a given model family is downloaded and registered."""

    config = TYPE_CONFIGS.get(model_type)
    if config is None:
        raise ValueError(f"Unknown model type '{model_type}'.")

    repo = get_model_repository(model_type)

    if model_type in _BOOTSTRAPPED_TYPES:
        return repo

    with _BOOTSTRAP_LOCK:
        repo = get_model_repository(model_type)

        resolved_repo_id = (
            repo_id or os.getenv(config.repo_env) or config.default_repo_id
        )
        if not resolved_repo_id:
            logger.warning(
                "Model repository for '%s' not configured. Set %s to enable "
                "automatic downloads.",
                model_type,
                config.repo_env,
            )
            return repo

        resolved_revision = (
            revision
            or os.getenv(config.revision_env)
            or config.default_revision
            or "main"
        )
        resolved_subdir = subdir or (
            os.getenv(config.subdir_env)
            if config.subdir_env and os.getenv(config.subdir_env) is not None
            else config.default_subdir
        )

        snapshot_path = Path(
            HFInterface().download_models(
                repo_id=resolved_repo_id, revision=resolved_revision
            )
        )

        _register_models(model_type, snapshot_path, resolved_subdir, config)
        _BOOTSTRAPPED_TYPES.add(model_type)
        return repo


def _register_models(
    model_type: str,
    snapshot_path: Path,
    subdir: Optional[str],
    config: ModelTypeConfig,
) -> None:
    repo = get_model_repository(model_type)
    root = snapshot_path / subdir if subdir else snapshot_path
    if not root.exists():
        logger.warning(
            "Model directory '%s' not found in snapshot %s", subdir, snapshot_path
        )
        return

    for config_path in _iter_config_paths(root, config.config_glob):
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Skipping %s: %s", config_path, exc)
            continue

        env_config = config.env_extractor(config_data)
        if not env_config:
            continue

        try:
            topology_hash = config.hash_fn(env_config)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Failed to compute topology hash for %s: %s", config_path, exc
            )
            continue

        if topology_hash in repo:
            continue

        model_path = _find_model_weights(config_path)
        coupling_map = extract_coupling_edges(env_config)

        try:
            model = config.loader(config_path, model_path)
        except Exception as exc:  # pragma: no cover - dependent on external files
            logger.warning("Failed to load model from %s: %s", config_path, exc)
            continue

        repo.register(topology_hash, model, coupling_map)


def _iter_config_paths(root: Path, pattern: str) -> Iterable[Path]:
    return root.rglob(pattern)


def _find_model_weights(config_path: Path) -> Optional[Path]:
    candidates = []
    stem = config_path.stem
    parent = config_path.parent

    for suffix in (".safetensors", ".pt", ".bin", ".ckpt", ".pth"):
        candidate = parent / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
        candidates.extend(parent.glob(f"{stem}*{suffix}"))

    return candidates[0] if candidates else None

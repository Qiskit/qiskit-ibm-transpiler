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

import yaml

from qiskit_gym.rl.synthesis import RLSynthesis

from .hf_models_client import HFInterface
from .model_repository import RLSynthesisRepository
from .utils import compute_topology_hash, extract_coupling_edges

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
        data_files = resources.files("qiskit_ibm_transpiler") / "data"
        yaml_file = data_files / "model_sources.yaml"
        content = yaml_file.read_text(encoding="utf-8")
        return yaml.safe_load(content) or {}
    except FileNotFoundError:
        logger.debug(
            "model_sources.yaml not found; falling back to env-only configuration"
        )
    except ModuleNotFoundError:
        logger.debug(
            "Data package not available; falling back to env-only configuration"
        )
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse YAML model sources: %s", exc)
    except Exception as exc:  # pragma: no cover - unexpected
        logger.warning("Failed to load static model sources: %s", exc)
    return {}


_STATIC_SOURCES = _load_static_sources()


def _normalize_type_name(model_type: str) -> str:
    cleaned = [c if c.isalnum() else "_" for c in model_type.upper()]
    return "".join(cleaned)


def _build_type_config(model_type: str) -> ModelTypeConfig:
    defaults = _STATIC_SOURCES.get(model_type, {})
    env_prefix = f"QISKIT_TRANSPILER_{_normalize_type_name(model_type)}"
    return ModelTypeConfig(
        repo_env=f"{env_prefix}_REPO_ID",
        revision_env=f"{env_prefix}_REVISION",
        subdir_env=f"{env_prefix}_SUBDIR",
        default_repo_id=defaults.get("repo_id"),
        default_revision=defaults.get("revision", "main"),
        default_subdir=defaults.get("subdir"),
    )


TYPE_CONFIGS: Dict[str, ModelTypeConfig] = {
    model_type: _build_type_config(model_type) for model_type in _STATIC_SOURCES.keys()
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
        TYPE_CONFIGS[model_type] = _build_type_config(model_type)
        config = TYPE_CONFIGS[model_type]

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

        try:
            snapshot_path = Path(
                HFInterface().download_models(
                    repo_id=resolved_repo_id, revision=resolved_revision
                )
            )
        except Exception as exc:
            logger.error(
                "Failed to download %s models from %s@%s. "
                "Check network connection and verify repository exists. "
                "To use a different repository, set %s. Error: %s",
                model_type,
                resolved_repo_id,
                resolved_revision,
                config.repo_env,
                exc,
            )
            return repo

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
        env_prefix = f"QISKIT_TRANSPILER_{_normalize_type_name(model_type)}"
        if subdir:
            logger.error(
                "Model subdirectory '%s' not found in downloaded snapshot at %s. "
                "Check that %s_SUBDIR is correct. Available directories: %s",
                subdir,
                snapshot_path,
                env_prefix,
                [d.name for d in snapshot_path.iterdir() if d.is_dir()],
            )
        else:
            logger.error(
                "No models found in snapshot at %s. "
                "The repository may be empty or invalid. "
                "Verify %s_REPO_ID points to a valid model repository.",
                snapshot_path,
                env_prefix,
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
            logger.debug("Failed to compute topology hash for %s: %s", config_path, exc)
            continue

        if topology_hash in repo:
            continue

        model_path = _find_model_weights(config_path)
        coupling_map = extract_coupling_edges(env_config)

        try:
            model = config.loader(config_path, model_path)
        except Exception as exc:  # pragma: no cover - dependent on external files
            logger.warning(
                "Failed to load %s model from %s: %s. "
                "The model file may be corrupted or incompatible. "
                "Try clearing the cache: rm -rf ~/.cache/huggingface/hub/models--*ai-transpiler*",
                model_type,
                config_path,
                exc,
            )
            continue

        repo.register(topology_hash, model, coupling_map)

    # Warn if no models were registered
    if len(repo) == 0:
        env_prefix = f"QISKIT_TRANSPILER_{_normalize_type_name(model_type)}"
        logger.warning(
            "No %s models were successfully registered. "
            "Synthesis for this model type will fail. "
            "Verify that %s_REPO_ID contains valid model files.",
            model_type,
            env_prefix,
        )


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

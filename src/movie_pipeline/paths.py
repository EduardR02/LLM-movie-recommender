"""Filesystem path helpers for the movie pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    """Collection of filesystem locations used by the pipeline."""

    root: Path
    data: Path
    imdb: Path
    artifacts: Path
    plots: Path
    analyses: Path
    embeddings: Path
    plot_embeddings: Path
    state: Path
    logs: Path
    cache: Path


def discover_paths() -> PipelinePaths:
    """Infer repository-relative directories and ensure they exist."""
    root = Path(__file__).resolve().parents[2]

    data = root / "data"
    imdb = data / "imdb"
    artifacts = root / "artifacts"
    plots = artifacts / "plots"
    analyses = artifacts / "grok"
    embeddings = artifacts / "embeddings"
    plot_embeddings = artifacts / "plot_embeddings"
    state = root / "state"
    logs = root / "logs"
    cache = root / "cache"

    for path in (
        data,
        imdb,
        artifacts,
        plots,
        analyses,
        embeddings,
        plot_embeddings,
        state,
        logs,
        cache,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return PipelinePaths(
        root=root,
        data=data,
        imdb=imdb,
        artifacts=artifacts,
        plots=plots,
        analyses=analyses,
        embeddings=embeddings,
        plot_embeddings=plot_embeddings,
        state=state,
        logs=logs,
        cache=cache,
    )


PATHS = discover_paths()


def _ensure_profile_dir(base: Path, profile: str) -> Path:
    normalized = (profile or "").strip()
    if not normalized:
        raise ValueError("Profile is required for profile-specific paths.")
    target = base if normalized == "default" else base / normalized
    target.mkdir(parents=True, exist_ok=True)
    return target


def analyses_dir(profile: str | None = None) -> Path:
    """Return the directory for Grok analyses for the given profile."""
    if profile is None or not str(profile).strip():
        raise ValueError("Profile is required for analyses_dir.")
    return _ensure_profile_dir(PATHS.analyses, str(profile))


def embeddings_dir(profile: str | None = None, variant: str | None = None) -> Path:
    """Return the directory for analysis embeddings for the given profile and variant."""
    if profile is None or not str(profile).strip():
        raise ValueError("Profile is required for embeddings_dir.")
    base = _ensure_profile_dir(PATHS.embeddings, str(profile))
    normalized_variant = (variant or "default").strip() or "default"
    if normalized_variant in {"", "default"}:
        return base
    target = base / normalized_variant
    target.mkdir(parents=True, exist_ok=True)
    return target

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
    normalized = (profile or "default").strip()
    if normalized in ("", "default"):
        base.mkdir(parents=True, exist_ok=True)
        return base
    target = base / normalized
    target.mkdir(parents=True, exist_ok=True)
    return target


def analyses_dir(profile: str | None = None) -> Path:
    """Return the directory for Grok analyses for the given profile."""
    return _ensure_profile_dir(PATHS.analyses, profile or "default")


def embeddings_dir(profile: str | None = None) -> Path:
    """Return the directory for analysis embeddings for the given profile."""
    return _ensure_profile_dir(PATHS.embeddings, profile or "default")

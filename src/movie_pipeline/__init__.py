"""Pipeline utilities for generating movie vibe embeddings."""

from __future__ import annotations

from importlib import import_module, metadata
from typing import Any

__all__ = [
    "AnalysisConfig",
    "AnalysisRunner",
    "EmbeddingConfig",
    "EmbeddingRunner",
    "IMDbConfig",
    "Manifest",
    "PlotFetcher",
    "WikipediaConfig",
    "ingest_imdb",
    "get_version",
]

_LAZY_ATTRS = {
    "AnalysisConfig": ("movie_pipeline.analysis", "AnalysisConfig"),
    "AnalysisRunner": ("movie_pipeline.analysis", "AnalysisRunner"),
    "EmbeddingConfig": ("movie_pipeline.embeddings", "EmbeddingConfig"),
    "EmbeddingRunner": ("movie_pipeline.embeddings", "EmbeddingRunner"),
    "IMDbConfig": ("movie_pipeline.imdb", "IMDbConfig"),
    "Manifest": ("movie_pipeline.manifest", "Manifest"),
    "PlotFetcher": ("movie_pipeline.plots", "PlotFetcher"),
    "WikipediaConfig": ("movie_pipeline.plots", "WikipediaConfig"),
    "ingest_imdb": ("movie_pipeline.imdb", "ingest_imdb"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_version() -> str:
    """Return the installed package version."""
    try:
        return metadata.version("movie-recommending")
    except metadata.PackageNotFoundError:
        return "0.0.0"

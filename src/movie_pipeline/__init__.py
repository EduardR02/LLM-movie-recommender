"""Pipeline utilities for generating movie vibe embeddings."""

from importlib import metadata

from .analysis import AnalysisConfig, AnalysisRunner
from .embeddings import EmbeddingConfig, EmbeddingRunner
from .imdb import IMDbConfig, ingest_imdb
from .manifest import Manifest
from .plots import PlotFetcher, WikipediaConfig

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


def get_version() -> str:
    """Return the installed package version."""
    try:
        return metadata.version("movie-recommending")
    except metadata.PackageNotFoundError:
        return "0.0.0"

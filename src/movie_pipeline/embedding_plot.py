"""Quick scatter plot for embedding clusters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from collections import Counter

from .embeddings import load_embeddings_matrix
from .manifest import Manifest
from .paths import PATHS, embeddings_dir


def _resolve_index_dir(index: str, profile: str) -> Path:
    if index == "analysis":
        return embeddings_dir(profile)
    if index == "plot":
        PATHS.plot_embeddings.mkdir(parents=True, exist_ok=True)
        return PATHS.plot_embeddings
    raise ValueError(f"Unknown index '{index}'. Use 'analysis' or 'plot'.")


def _sample_matrix(
    matrix: np.ndarray,
    ids: list[str],
    sample: int | None,
    seed: int,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    if sample is None or sample >= len(ids):
        return matrix, ids, None
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ids), size=sample, replace=False)
    return matrix[indices], [ids[i] for i in indices], indices


def _reduce_dimensions(matrix: np.ndarray, method: str) -> np.ndarray:
    if matrix.shape[1] < 2:
        raise ValueError("Embeddings have fewer than 2 dimensions; cannot plot.")
    if method == "first2":
        return matrix[:, :2]
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:  # noqa: NPY002
        raise RuntimeError(f"SVD failed while computing PCA: {exc}") from exc
    components = vh[:2].T
    return centered @ components


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m movie_pipeline.embedding_plot",
        description="Scatter plot of embedding vectors for quick clustering inspection.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Profile whose embeddings to visualize (default: default).",
    )
    parser.add_argument(
        "--index",
        default="analysis",
        choices=("analysis", "plot"),
        help="Embedding set to load (analysis or plot).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample this many embeddings before plotting.",
    )
    parser.add_argument(
        "--method",
        choices=("pca", "first2"),
        default="pca",
        help="Dimensionality reduction method (pca or use first two coordinates).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the plot (PNG, PDF, etc.).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    directory = _resolve_index_dir(args.index, args.profile)
    ids, matrix, _ = load_embeddings_matrix(directory)
    if matrix.size == 0:
        print(f"No embeddings found in {directory}.", file=sys.stderr)
        return 1

    matrix = matrix.astype(np.float32)
    matrix, ids, _ = _sample_matrix(matrix, ids, args.sample, args.seed)

    coords = _reduce_dimensions(matrix, args.method)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting embeddings.") from exc

    manifest = Manifest(profile=args.profile)
    manifest_ids = ids
    manifest_records = {record.tconst: record for record in manifest.iter_titles()}
    ratings = []
    primary_genres: list[str | None] = []
    for tconst in manifest_ids:
        record = manifest_records.get(tconst)
        if record:
            ratings.append(record.average_rating)
            genre = (record.genres or "").split(",")[0].strip() or None
            primary_genres.append(genre)
        else:
            ratings.append(np.nan)
            primary_genres.append(None)

    ratings = np.array(ratings, dtype=float)
    finite_mask = np.isfinite(ratings)
    if np.any(finite_mask):
        valid_ratings = ratings[finite_mask]
        min_rating = valid_ratings.min()
        max_rating = valid_ratings.max()
        if max_rating != min_rating:
            norm_ratings = (ratings - min_rating) / (max_rating - min_rating)
            norm_ratings = np.clip(norm_ratings, 0, 1)
        else:
            norm_ratings = None
    else:
        norm_ratings = None

    cmap_ratings = plt.get_cmap("viridis")
    if norm_ratings is not None:
        rating_colors = cmap_ratings(np.nan_to_num(norm_ratings, nan=0.5))
    else:
        default_rating_color = np.array([0.2, 0.4, 0.9, 1.0])
        rating_colors = np.tile(default_rating_color, (len(ids), 1))

    genre_counts = Counter(g for g in primary_genres if g)
    top_genres = [genre for genre, _ in genre_counts.most_common(20)]
    cmap_genres = plt.get_cmap("tab20", max(len(top_genres), 1))
    genre_color_map = {genre: np.array(cmap_genres(i % cmap_genres.N)) for i, genre in enumerate(top_genres)}
    default_genre_color = np.array([0.3, 0.3, 0.3, 1.0])
    genre_colors = np.vstack(
        [
            genre_color_map.get(genre, default_genre_color)
            if genre_color_map.get(genre, default_genre_color).shape[0] == 4
            else np.append(genre_color_map.get(genre, default_genre_color), 1.0)
            for genre in primary_genres
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    ax_rating, ax_genre = axes

    if norm_ratings is not None:
        order_rating = np.argsort(np.nan_to_num(norm_ratings, nan=-1))
        coords_ordered = coords[order_rating]
        rating_colors_ordered = rating_colors[order_rating]
    else:
        coords_ordered = coords
        rating_colors_ordered = rating_colors

    rating_scatter = ax_rating.scatter(
        coords_ordered[:, 0],
        coords_ordered[:, 1],
        s=12,
        alpha=0.7,
        c=rating_colors_ordered,
        edgecolors="none",
        linewidths=0,
    )
    ax_rating.set_title(f"IMDb rating – index={args.index} profile={args.profile} (n={len(ids)})")
    ax_rating.set_xlabel("Component 1" if args.method == "pca" else "Dimension 1")
    ax_rating.set_ylabel("Component 2" if args.method == "pca" else "Dimension 2")
    if norm_ratings is not None:
        cbar = fig.colorbar(rating_scatter, ax=ax_rating, pad=0.01, fraction=0.046)
        cbar.ax.set_ylabel("IMDb rating", rotation=270, labelpad=12)

    genre_frequencies = np.array([genre_counts.get(g, 0) for g in primary_genres])
    genre_order = np.argsort(genre_frequencies)
    coords_genre_ordered = coords[genre_order]
    genre_colors_ordered = genre_colors[genre_order]

    ax_genre.scatter(
        coords_genre_ordered[:, 0],
        coords_genre_ordered[:, 1],
        s=12,
        alpha=0.7,
        c=genre_colors_ordered,
        edgecolors="none",
        linewidths=0,
    )
    ax_genre.set_title("Primary genre")
    ax_genre.set_xlabel("Component 1" if args.method == "pca" else "Dimension 1")

    if top_genres:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=genre_color_map[genre],
                markersize=6,
                label=genre,
            )
            for genre in top_genres
        ]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=(0.3, 0.3, 0.3),
                markersize=6,
                label="Other/Unknown",
            )
        )
        ax_genre.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="small",
            title="Primary genre",
            borderaxespad=0.5,
        )
    else:
        ax_genre.text(
            0.5,
            0.5,
            "No genre data",
            transform=ax_genre.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    plt.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=200)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

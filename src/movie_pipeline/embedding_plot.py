"""Quick scatter plot for embedding clusters."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
from collections import Counter

from .embeddings import load_embeddings_matrix
from .manifest import Manifest
from .paths import PATHS, embeddings_dir

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
PLOTLY_TEMPLATE = TEMPLATES_DIR / "plotly_embedding.html"


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


def _rgba_to_css(rgba: np.ndarray) -> str:
    r = int(round(float(np.clip(rgba[0], 0.0, 1.0) * 255)))
    g = int(round(float(np.clip(rgba[1], 0.0, 1.0) * 255)))
    b = int(round(float(np.clip(rgba[2], 0.0, 1.0) * 255)))
    a = float(np.clip(rgba[3], 0.0, 1.0))
    return f"rgba({r}, {g}, {b}, {a:.3f})"


def _compute_axis_range(values: np.ndarray, padding: float) -> tuple[float, float]:
    padding = max(padding, 1.0)
    col_min = float(values.min())
    col_max = float(values.max())
    center = (col_max + col_min) / 2.0
    half_span = (col_max - col_min) / 2.0
    if half_span == 0.0:
        half_span = 1.0
    half_span *= padding
    return center - half_span, center + half_span


def _write_plotly_page(
    *,
    fig,
    search_data: list[dict[str, str | int]],
    page_title: str,
    div_id: str,
    output_path: Path | None,
    auto_open: bool,
    selection_info: dict[str, int],
) -> Path:
    if not PLOTLY_TEMPLATE.exists():
        raise RuntimeError(f"Plotly template missing at {PLOTLY_TEMPLATE}")

    template = PLOTLY_TEMPLATE.read_text(encoding="utf-8")
    figure_json = fig.to_json()
    figure_json_js = json.dumps(figure_json)
    config = {
        "responsive": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }
    config_json = json.dumps(config)
    search_json = json.dumps(search_data)
    selection_json = json.dumps(selection_info)

    html = (
        template.replace("{{FIGURE_JSON}}", figure_json_js)
        .replace("{{CONFIG_JSON}}", config_json)
        .replace("{{SEARCH_DATA}}", search_json)
        .replace("{{SELECTION_INFO}}", selection_json)
        .replace("{{DIV_ID}}", div_id)
        .replace("{{PAGE_TITLE}}", page_title)
    )

    if output_path is None:
        safe_profile = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in page_title
        )
        default_name = f"embedding-{safe_profile or 'plot'}.html"
        target_path = PATHS.plots / default_name
    else:
        target_path = output_path
        if target_path.suffix.lower() not in {".html", ".htm"}:
            target_path = target_path.with_suffix(".html")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(html, encoding="utf-8")

    if auto_open:
        try:
            webbrowser.open(target_path.as_uri(), new=2)
        except Exception:
            pass
    return target_path


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
        help="Optional path to save the plot (PNG/PDF for matplotlib, HTML for plotly).",
    )
    parser.add_argument(
        "--backend",
        choices=("plotly", "matplotlib"),
        default="plotly",
        help="Rendering backend for the scatter plot (default: plotly).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=18.0,
        help="Marker size for scatter points (default: 18).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Point transparency for scatter plots (default: 0.75).",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=1.5,
        help="Scale factor applied to PCA coordinates to separate clusters (default: 1.5).",
    )
    parser.add_argument(
        "--hover-info",
        action="store_true",
        help="Enable interactive hover tooltips (matplotlib backend only).",
    )
    parser.add_argument(
        "--fig-width",
        type=int,
        default=1200,
        help="Figure width in pixels (default: 1200).",
    )
    parser.add_argument(
        "--fig-height",
        type=int,
        default=650,
        help="Figure height in pixels (default: 650).",
    )
    parser.add_argument(
        "--axis-padding",
        type=float,
        default=1.05,
        help="Multiplier to expand x/y ranges for breathing room (default: 1.05).",
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
    coords = coords * float(args.spread)

    manifest = Manifest(profile=args.profile)
    manifest_records = {record.tconst: record for record in manifest.iter_titles()}
    ratings: list[float | np.floating] = []
    title_texts: list[str] = []
    primary_genres: list[str | None] = []
    for tconst in ids:
        record = manifest_records.get(tconst)
        if not record:
            ratings.append(np.nan)
            primary_genres.append(None)
            title_texts.append(tconst)
            continue
        ratings.append(record.average_rating)
        primary_genres.append((record.genres or "").split(",")[0].strip() or None)
        title_texts.append(record.primary_title)

    ratings = np.array(ratings, dtype=float)
    finite_mask = np.isfinite(ratings)
    if np.any(finite_mask):
        valid_ratings = ratings[finite_mask]
        min_rating = float(valid_ratings.min())
        max_rating = float(valid_ratings.max())
        if max_rating != min_rating:
            norm_ratings = (ratings - min_rating) / (max_rating - min_rating)
            norm_ratings = np.clip(norm_ratings, 0, 1)
        else:
            norm_ratings = None
    else:
        norm_ratings = None
        min_rating = max_rating = None

    genre_counts = Counter(g for g in primary_genres if g)
    top_genres = [genre for genre, _ in genre_counts.most_common(20)]

    hover_texts: list[str] = []
    search_entries: list[dict[str, str | int | float]] = []
    for index, (tconst, genre, label, rating_value) in enumerate(
        zip(ids, primary_genres, title_texts, ratings)
    ):
        record = manifest_records.get(tconst)
        genre_label = genre or "Unknown"
        if record:
            year = record.start_year or "????"
            if isinstance(rating_value, (int, float, np.floating)) and np.isfinite(rating_value):
                rating_str = f"{float(rating_value):.1f}"
            else:
                rating_str = "n/a"
            hover_texts.append(f"{record.primary_title} ({year})<br>IMDb {rating_str} · {genre_label}")
        else:
            hover_texts.append(tconst)
        search_entries.append(
            {
                "title": label,
                "id": tconst,
                "lower": label.lower(),
                "hover": hover_texts[-1],
                "x": float(coords[index, 0]),
                "y": float(coords[index, 1]),
            }
        )

    axis_padding = max(float(args.axis_padding), 1.0)
    x_range = _compute_axis_range(coords[:, 0], axis_padding)
    y_range = _compute_axis_range(coords[:, 1], axis_padding)
    point_size = max(float(args.point_size), 1.0)
    alpha = min(max(float(args.alpha), 0.05), 1.0)
    fig_width_px = max(int(args.fig_width), 600)
    fig_height_px = max(int(args.fig_height), 400)

    if args.backend == "plotly":
        try:
            import plotly.graph_objects as go
            from plotly import colors as plotly_colors
            from plotly.subplots import make_subplots
        except ModuleNotFoundError as exc:
            raise RuntimeError("Plotly is required for backend=plotly. Install plotly>=5.") from exc

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_xaxes=True,
            shared_yaxes=True,
            column_widths=[0.5, 0.5],
            subplot_titles=(
                f"IMDb rating – {args.index}/{args.profile}",
                "Primary genre",
            ),
            horizontal_spacing=0.05,
        )

        if np.any(finite_mask):
            rating_color_values = np.nan_to_num(ratings, nan=float(valid_ratings.mean()))
            cmin = min_rating
            cmax = max_rating
        else:
            rating_color_values = None
            cmin = cmax = None

        rating_trace = go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=point_size,
                opacity=alpha,
                color=rating_color_values if rating_color_values is not None else "#4a83f7",
                colorscale="Viridis" if rating_color_values is not None else None,
                colorbar=(
                    dict(title="IMDb rating", x=0.47, xanchor="left", len=0.85)
                    if rating_color_values is not None
                    else None
                ),
                cmin=cmin,
                cmax=cmax,
            ),
            hovertext=hover_texts,
            hoverinfo="text",
            name="Rating map",
            showlegend=False,
        )
        fig.add_trace(rating_trace, row=1, col=1)
        highlight_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers+text",
            marker=dict(
                size=point_size * 1.6,
                color="#ffd166",
                line=dict(color="#111", width=1.2),
                symbol="circle-open-dot",
            ),
            text=[],
            textfont=dict(color="#ffd166"),
            textposition="top center",
            name="Selection",
            hoverinfo="text",
            showlegend=False,
        )
        fig.add_trace(highlight_trace, row=1, col=1)
        highlight_trace_index = len(fig.data) - 1
        genre_highlight = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                size=point_size * 1.4,
                color="#ffd166",
                line=dict(color="#111", width=1.2),
                symbol="circle-open-dot",
            ),
            name="Selection (genre)",
            hoverinfo="skip",
            showlegend=False,
        )
        fig.add_trace(genre_highlight, row=1, col=2)
        genre_highlight_index = len(fig.data) - 1
        selection_info = {
            "rating_trace": 0,
            "highlight_trace": highlight_trace_index,
            "genre_highlight_trace": genre_highlight_index,
        }

        palette = (
            plotly_colors.qualitative.Plotly
            + plotly_colors.qualitative.Safe
            + plotly_colors.qualitative.Vivid
            + plotly_colors.qualitative.Dark24
            + plotly_colors.qualitative.Set3
        )
        max_genre_traces = min(len(top_genres), len(palette))
        highlighted_genres = top_genres[:max_genre_traces]
        genre_color_lookup = {
            genre: palette[idx % len(palette)] for idx, genre in enumerate(highlighted_genres)
        }
        default_genre_hex = "#7f7f7f"
        genre_bins: dict[str, list[int]] = {}
        for idx, genre in enumerate(primary_genres):
            label = genre if genre in genre_color_lookup else "Other/Unknown"
            if not label:
                label = "Other/Unknown"
            genre_bins.setdefault(label, []).append(idx)

        genre_trace_order = list(genre_color_lookup.keys())
        if "Other/Unknown" in genre_bins:
            genre_trace_order.append("Other/Unknown")

        for label in genre_trace_order:
            indices = genre_bins.get(label)
            if not indices:
                continue
            idx_array = np.array(indices, dtype=int)
            fig.add_trace(
                go.Scattergl(
                    x=coords[idx_array, 0],
                    y=coords[idx_array, 1],
                    mode="markers",
                    marker=dict(
                        size=point_size,
                        opacity=alpha,
                        color=genre_color_lookup.get(label, default_genre_hex),
                    ),
                    hovertext=[hover_texts[i] for i in idx_array],
                    hoverinfo="text",
                    name=label,
                    legendgroup=label,
                    showlegend=True,
                ),
                row=1,
                col=2,
            )

        axis_label = "Component" if args.method == "pca" else "Dimension"
        fig.update_xaxes(title_text=f"{axis_label} 1", range=x_range, row=1, col=1)
        fig.update_yaxes(title_text=f"{axis_label} 2", range=y_range, row=1, col=1)
        fig.update_xaxes(title_text=f"{axis_label} 1", range=x_range, matches="x", row=1, col=2)
        fig.update_yaxes(title_text=f"{axis_label} 2", range=y_range, matches="y", row=1, col=2)

        fig.update_layout(
            title=f"Embedding explorer (n={len(ids)})",
            height=fig_height_px,
            width=fig_width_px,
            legend=dict(
                title="Genre",
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            margin=dict(l=60, r=280, t=90, b=60),
            hovermode="closest",
        )

        page_title = f"{args.profile} · {args.index} · {args.method.upper()}"
        div_id = "embedding-plot"
        output_path = args.output
        auto_open = output_path is None
        target = _write_plotly_page(
            fig=fig,
            search_data=search_entries,
            page_title=page_title,
            div_id=div_id,
            output_path=output_path,
            auto_open=auto_open,
            selection_info=selection_info,
        )
        print(f"Interactive plot saved to {target}")
        if auto_open:
            print("Opened in your browser; use the search box to jump to any title.")
        return 0

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for backend=matplotlib.") from exc

    cmap_ratings = plt.get_cmap("viridis")
    if norm_ratings is not None:
        rating_colors = cmap_ratings(np.nan_to_num(norm_ratings, nan=0.5))
    else:
        default_rating_color = np.array([0.2, 0.4, 0.9, 1.0])
        rating_colors = np.tile(default_rating_color, (len(ids), 1))

    cmap_genres = plt.get_cmap("tab20", max(len(top_genres), 1))
    genre_color_map = {
        genre: np.array(cmap_genres(i % cmap_genres.N)) for i, genre in enumerate(top_genres)
    }
    default_genre_color = np.array([0.3, 0.3, 0.3, 1.0])
    genre_color_rows = []
    for genre in primary_genres:
        color = genre_color_map.get(genre, default_genre_color)
        if color.shape[0] != 4:
            color = np.append(color[:3], 1.0)
        genre_color_rows.append(color)
    genre_colors = np.vstack(genre_color_rows)

    dpi = 96
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_width_px / dpi, fig_height_px / dpi),
        sharex=True,
        sharey=True,
    )
    ax_rating, ax_genre = axes

    if norm_ratings is not None:
        order_rating = np.argsort(np.nan_to_num(norm_ratings, nan=-1))
    else:
        order_rating = np.arange(len(ids))
    coords_ordered = coords[order_rating]
    rating_colors_ordered = rating_colors[order_rating]

    rating_scatter = ax_rating.scatter(
        coords_ordered[:, 0],
        coords_ordered[:, 1],
        s=point_size,
        alpha=alpha,
        c=rating_colors_ordered,
        edgecolors="none",
        linewidths=0,
    )
    ax_rating.set_title(f"IMDb rating – index={args.index} profile={args.profile} (n={len(ids)})")
    ax_rating.set_xlabel("Component 1" if args.method == "pca" else "Dimension 1")
    ax_rating.set_ylabel("Component 2" if args.method == "pca" else "Dimension 2")
    ax_rating.set_xlim(x_range)
    ax_rating.set_ylim(y_range)
    if norm_ratings is not None:
        cbar = fig.colorbar(rating_scatter, ax=ax_rating, pad=0.01, fraction=0.046)
        cbar.ax.set_ylabel("IMDb rating", rotation=270, labelpad=12)

    genre_frequencies = np.array([genre_counts.get(g, 0) for g in primary_genres])
    genre_order = np.argsort(genre_frequencies)
    coords_genre_ordered = coords[genre_order]
    genre_colors_ordered = genre_colors[genre_order]

    genre_scatter = ax_genre.scatter(
        coords_genre_ordered[:, 0],
        coords_genre_ordered[:, 1],
        s=point_size,
        alpha=alpha,
        c=genre_colors_ordered,
        edgecolors="none",
        linewidths=0,
    )
    ax_genre.set_title("Primary genre")
    ax_genre.set_xlabel("Component 1" if args.method == "pca" else "Dimension 1")
    ax_genre.set_xlim(x_range)
    ax_genre.set_ylim(y_range)

    if args.hover_info:
        try:
            import mplcursors
        except ModuleNotFoundError:
            print("Install mplcursors to enable hover tooltips.", file=sys.stderr)
        else:
            def _format_tooltip(global_index: int) -> str:
                record = manifest_records.get(ids[global_index])
                if not record:
                    return ids[global_index]
                rating_value = (
                    f"{record.average_rating:.1f}" if record.average_rating is not None else "n/a"
                )
                year = record.start_year or "????"
                genre = (record.genres or "").split(",")[0].strip() or "Unknown"
                return f"{record.primary_title} ({year})\nIMDb {rating_value} · {genre}"

            def _attach_cursor(collection, ordering: np.ndarray):
                cursor = mplcursors.cursor(collection, hover=True)

                @cursor.connect("add")
                def _on_add(selection):
                    try:
                        local_index = int(selection.index)
                    except AttributeError:
                        selection.annotation.set_text("Unavailable")
                        return
                    point_index = int(ordering[local_index])
                    selection.annotation.set_text(_format_tooltip(point_index))
                    selection.annotation.get_bbox_patch().set(fc="white", alpha=0.92)

            _attach_cursor(rating_scatter, order_rating)
            _attach_cursor(genre_scatter, genre_order)

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

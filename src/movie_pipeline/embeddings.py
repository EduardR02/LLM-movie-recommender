"""Embedding generation and similarity search."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from openai import OpenAI

from .manifest import EmbeddingCandidate, Manifest, TitleRecord
from .paths import PATHS, embeddings_dir

LOGGER = logging.getLogger(__name__)


def save_embedding_vector(tconst: str, vector: list[float] | "np.ndarray", directory: Path) -> Path:
    """Persist an embedding vector to the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    import numpy as np

    path = directory / f"{tconst}.npy"
    arr = np.asarray(vector, dtype=np.float32)
    np.save(path, arr, allow_pickle=False)
    return path


@dataclass
class EmbeddingConfig:
    """Configuration for embedding computation."""

    model: str = "text-embedding-3-large"
    batch_size: int = 32
    timeout_seconds: int = 60
    dry_run: bool = False
    dimensions: int | None = 1024


class OpenAIEmbeddingClient:
    """Minimal client for OpenAI's embedding endpoint."""

    def __init__(self, api_key: str, config: EmbeddingConfig) -> None:
        self.config = config
        self._client = OpenAI(api_key=api_key, timeout=config.timeout_seconds)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], dict[str, int]]:
        response = self._client.embeddings.create(
            model=self.config.model,
            input=texts,
            dimensions=self.config.dimensions,
        )
        embeddings = [item.embedding for item in response.data]
        usage = response.usage or {}
        usage_counts = {
            "total_tokens": int(getattr(usage, "total_tokens", 0)),
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0)),
        }
        return embeddings, usage_counts


class EmbeddingRunner:
    """Generate embeddings for Grok analyses and persist them."""

    def __init__(self, manifest: Manifest, config: EmbeddingConfig | None = None) -> None:
        self.manifest = manifest
        self.config = config or EmbeddingConfig()
        self.profile = getattr(self.manifest, "profile", "default")
        self.embeddings_dir = embeddings_dir(self.profile)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not self.config.dry_run:
            raise RuntimeError("OPENAI_API_KEY environment variable is required.")
        self.client = OpenAIEmbeddingClient(api_key or "dry-run-key", self.config)

    def run(self, *, limit: int | None = None, force: bool = False) -> None:
        candidates = list(
            self.manifest.iter_embedding_candidates(
                limit,
                include_completed=force,
                profile=self.profile,
            )
        )
        if not candidates:
            LOGGER.warning("No embedding candidates found.")
            return

        if force:
            LOGGER.info("Force flag enabled; regenerating embeddings for %d titles.", len(candidates))
            for candidate in candidates:
                self._prepare_for_regeneration(candidate.title.tconst)

        LOGGER.info("Generating embeddings for %d analyses", len(candidates))
        session_id = f"emb-{uuid.uuid4().hex[:10]}"
        component_name = f"openai_embeddings[{self.profile}]"
        self.manifest.register_session(session_id, component=component_name)

        total_tokens = 0
        for batch in batched(candidates, self.config.batch_size):
            texts = [self._read_analysis(candidate) for candidate in batch]
            ids = [candidate.title.tconst for candidate in batch]
            if self.config.dry_run:
                LOGGER.info("Dry run enabled; skipping embedding batch for %s", ids)
                continue
            try:
                embeddings, usage = self.client.embed(texts)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Embedding call failed for batch %s: %s", ids, exc)
                for candidate in batch:
                    self.manifest.update_embedding_status(
                        candidate.title.tconst,
                        "error",
                        profile=self.profile,
                        session_id=session_id,
                        error=str(exc),
                    )
                continue

            total_tokens += usage.get("total_tokens", 0)
            if len(embeddings) != len(batch):
                LOGGER.error(
                    "Embedding response size mismatch: expected %d, got %d",
                    len(batch),
                    len(embeddings),
                )
                for candidate in batch:
                    self.manifest.update_embedding_status(
                        candidate.title.tconst,
                        "error",
                        profile=self.profile,
                        session_id=session_id,
                        error="embedding_response_mismatch",
                    )
                continue

            per_doc_tokens = usage.get("prompt_tokens", 0) // max(len(batch), 1)
            for candidate, vector in zip(batch, embeddings, strict=True):
                vector_path = self._write_embedding(candidate.title.tconst, vector)
                self.manifest.update_embedding_status(
                    candidate.title.tconst,
                    "ok",
                    profile=self.profile,
                    session_id=session_id,
                    model=self.config.model,
                    dim=len(vector),
                    vector_path=str(vector_path),
                    input_tokens=per_doc_tokens,
                )

        self.manifest.finalize_session(
            session_id,
            total_input_tokens=total_tokens,
            total_cached_input_tokens=0,
            total_output_tokens=0,
            total_reasoning_tokens=0,
            total_cost=self._estimate_cost(total_tokens),
            notes=f"profile={self.profile} titles={len(candidates)}",
        )
        LOGGER.info("Embedding generation complete (%d titles)", len(candidates))

    def _read_analysis(self, candidate: EmbeddingCandidate) -> str:
        try:
            return candidate.analysis_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise RuntimeError(f"Analysis file not found at {candidate.analysis_path}")

    def _write_embedding(self, tconst: str, vector: list[float]) -> Path:
        return save_embedding_vector(tconst, vector, self.embeddings_dir)

    def _prepare_for_regeneration(self, tconst: str) -> None:
        """Remove any existing embedding artifacts and reset manifest status."""
        npy_path = self.embeddings_dir / f"{tconst}.npy"
        json_path = self.embeddings_dir / f"{tconst}.json"
        for path in (npy_path, json_path):
            if path.exists():
                path.unlink()
        self.manifest.update_embedding_status(
            tconst,
            "queued",
            profile=self.profile,
            session_id=None,
            model=None,
            prompt_hash=None,
            dim=0,
            input_tokens=0,
            vector_path="",
            error=None,
        )

    def _estimate_cost(self, tokens: int) -> float:
        # Placeholder pricing for embeddings: $0.13 per 1M tokens.
        return round((tokens / 1_000_000) * 0.13, 6)


def batched(items: Iterable[EmbeddingCandidate], size: int) -> Iterable[list[EmbeddingCandidate]]:
    batch: list[EmbeddingCandidate] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class PlotEmbeddingRunner:
    """Generate embeddings directly from cleaned plot texts."""

    def __init__(self, manifest: Manifest, config: EmbeddingConfig | None = None) -> None:
        self.manifest = manifest
        self.config = config or EmbeddingConfig()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not self.config.dry_run:
            raise RuntimeError("OPENAI_API_KEY environment variable is required.")
        self.client = OpenAIEmbeddingClient(api_key or "dry-run-key", self.config)

    def run(self, *, limit: int | None = None, force: bool = False) -> None:
        candidates = self._collect_candidates(limit=limit, force=force)
        if not candidates:
            LOGGER.warning("No plot embedding candidates found.")
            return

        LOGGER.info("Generating plot embeddings for %d titles", len(candidates))
        session_id = f"plt-{uuid.uuid4().hex[:10]}"
        self.manifest.register_session(session_id, component="openai_plot_embeddings")

        total_tokens = 0
        processed = 0
        for batch in _batched_plot_candidates(candidates, self.config.batch_size):
            texts = [text for _, text in batch]
            records = [candidate for candidate, _ in batch]
            ids = [candidate.title.tconst for candidate in records]

            if self.config.dry_run:
                LOGGER.info("Dry run enabled; skipping plot embedding batch for %s", ids)
                continue

            try:
                embeddings, usage = self.client.embed(texts)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Plot embedding call failed for batch %s: %s", ids, exc)
                continue

            total_tokens += usage.get("total_tokens", 0)
            for candidate, vector in zip(records, embeddings, strict=True):
                save_embedding_vector(candidate.title.tconst, vector, PATHS.plot_embeddings)
                processed += 1

        self.manifest.finalize_session(
            session_id,
            total_input_tokens=total_tokens,
            total_cached_input_tokens=0,
            total_output_tokens=0,
            total_reasoning_tokens=0,
            total_cost=self._estimate_cost(total_tokens),
            notes=f"titles={processed}",
        )
        LOGGER.info("Plot embedding generation complete (%d titles)", processed)

    def _collect_candidates(self, *, limit: int | None, force: bool) -> list[PlotEmbeddingCandidate]:
        candidates: list[PlotEmbeddingCandidate] = []
        target_count = limit if limit is not None else None
        for record in self.manifest.iter_titles():
            plot_path = PATHS.plots / f"{record.tconst}.txt"
            if not plot_path.exists():
                continue
            vector_path = PATHS.plot_embeddings / f"{record.tconst}.npy"
            if not force and vector_path.exists():
                continue
            if force and vector_path.exists():
                vector_path.unlink()
            candidates.append(PlotEmbeddingCandidate(title=record, plot_path=plot_path))
            if target_count is not None and len(candidates) >= target_count:
                break
        return candidates

    def _estimate_cost(self, tokens: int) -> float:
        return round((tokens / 1_000_000) * 0.13, 6)


def _batched_plot_candidates(
    candidates: list[PlotEmbeddingCandidate],
    batch_size: int,
):
    batch: list[tuple[PlotEmbeddingCandidate, str]] = []
    for candidate in candidates:
        try:
            text = candidate.plot_path.read_text(encoding="utf-8").strip()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to read plot for %s: %s", candidate.title.primary_title, exc)
            continue
        if not text:
            LOGGER.warning("Empty plot text for %s; skipping", candidate.title.primary_title)
            continue
        batch.append((candidate, text))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def load_embeddings_matrix(vectors_dir: Path | None = None):
    """Load embeddings and IDs into NumPy structures for similarity queries."""
    directory = vectors_dir or PATHS.embeddings
    files = sorted(list(directory.glob("*.npy")) + list(directory.glob("*.json")))
    ids: list[str] = []
    vectors: list = []
    dims: int | None = None
    for file in files:
        if file.suffix == ".npy":
            try:
                import numpy as np

                arr = np.load(file, allow_pickle=False)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to load embedding file %s: %s", file, exc)
                continue
            ids.append(file.stem)
            vectors.append(arr)
            dims = arr.shape[-1]
        else:
            try:
                payload = json.loads(file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                LOGGER.error("Failed to parse embedding file %s: %s", file, exc)
                continue
            ids.append(payload["tconst"])
            vectors.append(payload["embedding"])
            if dims is None:
                dims = len(payload["embedding"])
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required to load embeddings matrix.") from exc

    matrix = np.array(vectors, dtype=np.float32)
    return ids, matrix, dims or (matrix.shape[1] if matrix.size else 0)


def load_embedding_vector(tconst: str, vectors_dir: Path | None = None) -> tuple:
    """Load a single embedding vector for a title."""
    directory = vectors_dir or PATHS.embeddings
    npy_path = directory / f"{tconst}.npy"
    if npy_path.exists():
        try:
            import numpy as np

            arr = np.load(npy_path, allow_pickle=False).astype(np.float32)
            return arr, arr.shape[-1]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load embedding for {tconst}: {exc}") from exc

    json_path = directory / f"{tconst}.json"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed JSON embedding for {tconst}: {exc}") from exc
        vector = payload["embedding"]
        return vector, len(vector)

    raise FileNotFoundError(f"No embedding file found for {tconst}")


def top_k_similar(
    matrix,
    query_vector,
    ids: list[str],
    *,
    top_k: int = 10,
    largest: bool = True,
) -> list[tuple[str, float]]:
    """Return the top-k cosine similarities (or least similar when largest=False)."""
    import numpy as np

    query = np.array(query_vector, dtype=np.float32)
    scores = matrix @ query
    total = scores.shape[0]
    if top_k <= 0 or total == 0:
        return []
    limit = min(top_k, total)
    if largest:
        partition_scores = -scores
        sort_key = lambda item: -item[1]
    else:
        partition_scores = scores
        sort_key = lambda item: item[1]
    best_indices = np.argpartition(partition_scores, limit - 1)[:limit]
    best_pairs = ((ids[i], float(scores[i])) for i in best_indices)
    return sorted(best_pairs, key=sort_key)


def combine_embeddings(
    vectors: Sequence[Sequence[float]],
    *,
    weights: Sequence[float] | None = None,
    weight_tolerance: float = 1e-5,
):
    """Blend unit-length embeddings and return a normalized centroid."""
    import numpy as np

    if not vectors:
        raise ValueError("At least one embedding vector is required.")

    arrays: list[np.ndarray] = []
    filtered_weights: list[float] = []
    for index, vector in enumerate(vectors):
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            LOGGER.warning("Skipping zero-length embedding at index %d", index)
            continue
        arrays.append(arr / norm)
        if weights is not None:
            filtered_weights.append(weights[index])

    if not arrays:
        raise ValueError("All embedding vectors were zero-length; cannot combine.")

    stack = np.stack(arrays, axis=0)

    if weights is None:
        combined = stack.mean(axis=0)
    else:
        if len(weights) != len(vectors):
            raise ValueError(
                f"Expected {len(vectors)} weight(s) but received {len(weights)}."
            )
        if len(filtered_weights) != stack.shape[0]:
            raise ValueError(
                "Weights associated with zero-length embeddings cannot be applied."
            )
        weight_array = np.asarray(filtered_weights, dtype=np.float32)
        if (weight_array < 0).any():
            raise ValueError("Weights must be non-negative.")
        total = float(weight_array.sum())
        if total == 0.0:
            raise ValueError("Weights must sum to a positive value.")
        if not np.isclose(total, 1.0, atol=weight_tolerance):
            raise ValueError("Weights must sum to 1.0.")
        combined = np.average(stack, axis=0, weights=weight_array)

    combined_norm = float(np.linalg.norm(combined))
    if combined_norm == 0.0:
        raise ValueError("Combined embedding norm is zero.")
    return (combined / combined_norm).astype(np.float32)


@dataclass
class PlotEmbeddingCandidate:
    title: TitleRecord
    plot_path: Path

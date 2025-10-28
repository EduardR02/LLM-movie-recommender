"""FastAPI server that exposes the movie pipeline features over HTTP."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Iterator, Literal, Sequence, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .embeddings import (
    EmbeddingConfig,
    OpenAIEmbeddingClient,
    QwenEmbeddingClient,
    SentenceTransformerEmbeddingClient,
    combine_embeddings,
    resolve_embedding_instruction,
    is_qwen_embedding_model,
    intersection_similar,
    load_embeddings_matrix,
    top_k_similar,
)
from .explanations import (
    CandidateContext,
    SeedContext,
    SimilarityExplanationConfig,
    SimilarityExplainer,
)
from .manifest import Manifest, TitleRecord
from .paths import PATHS, analyses_dir, embeddings_dir

LOGGER = logging.getLogger(__name__)

EMBEDDING_INDEX_CHOICES = ("analysis", "plot")
TITLE_WITH_YEAR_RE = re.compile(r"^(?P<title>.+?)\s*\((?P<year>\d{4})\)$")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_profile(profile: str | None) -> str:
    value = (profile or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Profile is required.")
    return value


def _analysis_path(profile: str, tconst: str) -> Path:
    return analyses_dir(profile) / f"{tconst}.txt"


def _plot_path(tconst: str) -> Path:
    return PATHS.plots / f"{tconst}.txt"


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to read %s: %s", path, exc)
        return None


def _truncate_text(text: str | None, *, max_chars: int) -> str | None:
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    clipped = text[: max(0, max_chars - 3)].rstrip()
    return f"{clipped}..."


def _title_to_payload(record: TitleRecord) -> dict:
    return {
        "tconst": record.tconst,
        "primary_title": record.primary_title,
        "original_title": record.original_title,
        "title_type": record.title_type,
        "start_year": record.start_year,
        "end_year": record.end_year,
        "runtime_minutes": record.runtime_minutes,
        "genres": record.genres,
        "num_votes": record.num_votes,
        "average_rating": record.average_rating,
        "sort_rank": record.sort_rank,
    }


def _discover_profiles() -> list[str]:
    manifest = Manifest()
    profiles: set[str] = set()
    with manifest.cursor() as cur:
        for row in cur.execute(
            """
            SELECT DISTINCT profile FROM analyses
            UNION
            SELECT DISTINCT profile FROM embeddings
            ORDER BY profile ASC;
            """
        ):
            value = (row[0] or "").strip()
            if value:
                profiles.add(value)
    if profiles:
        manifest.close()
        return sorted(profiles)

    for base in (PATHS.analyses, PATHS.embeddings):
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir():
                name = child.name.strip()
                if name and not name.startswith("."):
                    profiles.add(name)
    manifest.close()
    return sorted(profiles)


def _embedding_meta_path(directory: Path) -> Path:
    return directory / ".cache" / "matrix.meta.json"


@dataclass
class EmbeddingCacheEntry:
    ids: list[str]
    matrix: object
    dimension: int
    directory: Path
    index: dict[str, int]
    meta_mtime_ns: int | None


_EMBEDDING_CACHE: dict[tuple[str, str, str], EmbeddingCacheEntry] = {}
LOCAL_EMBEDDING_PROVIDERS = {"sentence-transformers", "huggingface", "hf"}


@dataclass
class LocalClientEntry:
    client: SentenceTransformerEmbeddingClient | QwenEmbeddingClient
    config: EmbeddingConfig
    refcount: int
    last_used: float


class LocalModelRegistry:
    """Reference-counted cache for local embedding clients."""

    def __init__(self) -> None:
        self._entries: dict[Tuple[str, str, str, str, int, int], LocalClientEntry] = {}
        self._lock = Lock()

    def _key(self, config: EmbeddingConfig) -> Tuple[str, str, str, str, int, int]:
        provider = (config.provider or "sentence-transformers").strip().lower()
        model = (config.model or "").strip()
        device = (config.device or "").strip()
        instruction = (config.instruction or "").strip()
        dimensions = int(config.dimensions or 0)
        max_length = int(config.max_length or 0)
        return (provider, model, device, instruction, dimensions, max_length)

    def ensure(self, config: EmbeddingConfig, client: SentenceTransformerEmbeddingClient | QwenEmbeddingClient) -> None:
        key = self._key(config)
        with self._lock:
            entry = self._entries.get(key)
            if entry:
                entry.refcount += 1
                entry.last_used = time.time()
                return
            self._entries[key] = LocalClientEntry(
                client=client,
                config=config,
                refcount=1,
                last_used=time.time(),
            )

    def release(self, config: EmbeddingConfig) -> bool | None:
        key = self._key(config)
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None
            entry.refcount -= 1
            should_close = entry.refcount <= 0
            if should_close:
                self._entries.pop(key, None)
        if should_close:
            with contextlib.suppress(Exception):
                if hasattr(entry.client, "flush"):
                    entry.client.flush()
            with contextlib.suppress(Exception):
                if hasattr(entry.client, "close"):
                    entry.client.close()
        return should_close

    def get(self, config: EmbeddingConfig) -> SentenceTransformerEmbeddingClient | QwenEmbeddingClient | None:
        key = self._key(config)
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None
            entry.last_used = time.time()
            return entry.client


_LOCAL_MODEL_REGISTRY = LocalModelRegistry()


def _normalize_provider(provider: str | None, *, default: str) -> str:
    value = (provider or default).strip().lower()
    return value or default


@contextlib.contextmanager
def _embedding_client_session(
    config: EmbeddingConfig,
) -> Iterator[OpenAIEmbeddingClient | SentenceTransformerEmbeddingClient | QwenEmbeddingClient]:
    provider = _normalize_provider(config.provider, default="openai")
    if provider == "openai":
        client = _build_openai_client(config)
        try:
            yield client
        finally:
            with contextlib.suppress(Exception):
                if hasattr(client, "flush"):
                    client.flush()
        return

    if provider not in LOCAL_EMBEDDING_PROVIDERS:
        raise RuntimeError(f"Unsupported embedding provider '{config.provider}'.")

    client = _LOCAL_MODEL_REGISTRY.get(config)
    if client is None:
        client = _create_local_embedding_client(config)
    _LOCAL_MODEL_REGISTRY.ensure(config, client)
    try:
        yield client
    finally:
        with contextlib.suppress(Exception):
            if hasattr(client, "flush"):
                client.flush()
        _LOCAL_MODEL_REGISTRY.release(config)


def _load_embeddings(index: str, profile: str, embedding_set: str | None = None) -> EmbeddingCacheEntry:
    if index == "analysis":
        directory = embeddings_dir(profile, embedding_set)
        variant = (embedding_set or "default").strip() or "default"
    elif index == "plot":
        PATHS.plot_embeddings.mkdir(parents=True, exist_ok=True)
        directory = PATHS.plot_embeddings
        variant = "plot"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown embedding index '{index}'.")

    cache_key = (index, profile, variant)
    meta_path = _embedding_meta_path(directory)
    try:
        meta_mtime_ns = meta_path.stat().st_mtime_ns
    except FileNotFoundError:
        meta_mtime_ns = None

    entry = _EMBEDDING_CACHE.get(cache_key)
    if entry and entry.meta_mtime_ns == meta_mtime_ns:
        if entry.matrix.size == 0:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No embeddings found for index '{index}' (profile '{profile}', "
                    f"variant '{variant}')."
                ),
            )
        return entry

    ids, matrix, dims = load_embeddings_matrix(directory)
    if matrix.size == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No embeddings found for index '{index}' (profile '{profile}', "
                f"variant '{variant}')."
            ),
        )
    try:
        meta_mtime_ns = meta_path.stat().st_mtime_ns
    except FileNotFoundError:
        meta_mtime_ns = None

    index_map = {tconst: idx for idx, tconst in enumerate(ids)}
    entry = EmbeddingCacheEntry(
        ids=ids,
        matrix=matrix,
        dimension=dims,
        directory=directory,
        index=index_map,
        meta_mtime_ns=meta_mtime_ns,
    )
    _EMBEDDING_CACHE[cache_key] = entry
    return entry


_MANIFEST_CACHE: Dict[str, Manifest] = {}


def _manifest(profile: str) -> Manifest:
    normalized = _normalize_profile(profile)
    cached = _MANIFEST_CACHE.get(normalized)
    if cached:
        return cached
    manifest = Manifest(profile=normalized)
    _MANIFEST_CACHE[normalized] = manifest
    return manifest


def _resolve_titles(
    manifest: Manifest,
    identifiers: Sequence[str],
    *,
    matches: int,
) -> list[TitleRecord]:
    seed_records: list[TitleRecord] = []
    for identifier in identifiers:
        clean = identifier.strip()
        if not clean:
            continue
        if clean.lower().startswith("tt"):
            record = manifest.get_title(clean)
            if not record:
                raise HTTPException(status_code=404, detail=f"No title found for {clean}.")
            seed_records.append(record)
            continue
        search_title = clean
        target_year: int | None = None
        match = TITLE_WITH_YEAR_RE.match(clean)
        if match:
            search_title = match.group("title").strip()
            try:
                target_year = int(match.group("year"))
            except ValueError:
                target_year = None

        results = manifest.search_titles(search_title, limit=matches)
        if not results:
            raise HTTPException(status_code=404, detail=f"No titles matched '{clean}'.")

        normalized_search = search_title.casefold()
        exact_match = next(
            (
                record
                for record in results
                if record.primary_title.casefold() == normalized_search
            ),
            None,
        )
        if exact_match:
            if target_year and exact_match.start_year != target_year:
                exact_match = next(
                    (
                        record
                        for record in results
                        if record.primary_title.casefold() == normalized_search
                        and record.start_year == target_year
                    ),
                    exact_match,
                )
            seed_records.append(exact_match)
            continue

        if target_year:
            year_candidate = next(
                (record for record in results if record.start_year == target_year),
                None,
            )
            if year_candidate:
                seed_records.append(year_candidate)
                continue

        seed_records.append(results[0])
    if not seed_records:
        raise HTTPException(status_code=400, detail="Provide at least one seed title.")
    return seed_records


def _text_search_matches(
    *,
    target: Literal["analysis", "plot"],
    profile: str,
    query: str,
    limit: int,
) -> list[TextMatch]:
    manifest = _manifest(profile)
    directory = analyses_dir(profile) if target == "analysis" else PATHS.plots
    matches: list[TextMatch] = []
    needle = query.casefold()
    for path in sorted(directory.glob("*.txt")):
        if len(matches) >= limit:
            break
        text = _read_text(path)
        if not text:
            continue
        haystack = text.casefold()
        index = haystack.find(needle)
        if index == -1:
            continue
        start = max(0, index - 120)
        end = min(len(text), index + 120)
        snippet = text[start:end].strip()
        record = manifest.get_title(path.stem)
        matches.append(
            TextMatch(
                tconst=path.stem,
                primary_title=record.primary_title if record else None,
                start_year=record.start_year if record else None,
                snippet=snippet,
            )
        )
    return matches


def _format_label(record: TitleRecord) -> str:
    year = record.start_year or "????"
    return f"{record.primary_title} ({year})"


def _require_analysis(profile: str, tconst: str) -> str:
    text = _read_text(_analysis_path(profile, tconst))
    if not text:
        raise HTTPException(
            status_code=404,
            detail=f"No Grok analysis stored for {tconst}. Run the analysis pipeline first.",
        )
    return text


_EXPLAINER: SimilarityExplainer | None = None


def _get_explainer() -> SimilarityExplainer:
    global _EXPLAINER  # noqa: PLW0603
    if _EXPLAINER is None:
        _EXPLAINER = SimilarityExplainer(config=SimilarityExplanationConfig())
    return _EXPLAINER


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------


class TitleSummary(BaseModel):
    tconst: str
    primary_title: str
    original_title: str | None
    title_type: str
    start_year: int | None
    end_year: int | None
    runtime_minutes: int | None
    genres: str | None
    num_votes: int
    average_rating: float
    sort_rank: int


class Recommendation(BaseModel):
    title: TitleSummary
    score: float
    analysis_preview: str | None = None
    plot_preview: str | None = None


class EmbeddingSetInfo(BaseModel):
    name: str
    provider: str
    model: str | None = None
    dimension: int | None = None
    vector_count: int = 0
    updated_at: str | None = None


class EmbeddingSetResponse(BaseModel):
    embedding_sets: list[EmbeddingSetInfo]


class ManifestSummaryResponse(BaseModel):
    total_titles: int
    plot_count: int
    wikipedia_articles: int
    analysis_count: int
    embedding_count: int


class TitleQueryRequest(BaseModel):
    profile: str = Field(..., min_length=1)
    index: Literal["analysis", "plot"] = Field(default="analysis")
    embedding_set: str = Field(default="default")
    identifiers: list[str] = Field(min_items=1)
    top_k: int = Field(default=10, ge=1, le=100)
    matches: int = Field(default=5, ge=1, le=20)
    score_mode: Literal["centroid", "intersection"] = Field(default="centroid")
    least_similar: bool = False
    weights: list[float] | None = None
    include_analysis: bool = False
    include_plot: bool = False
    analysis_preview_chars: int = Field(default=480, ge=50, le=2000)
    plot_preview_chars: int = Field(default=480, ge=50, le=2000)


class TextQueryRequest(BaseModel):
    profile: str = Field(..., min_length=1)
    index: Literal["analysis", "plot"] = Field(default="analysis")
    text: str = Field(min_length=3, max_length=5000)
    top_k: int = Field(default=10, ge=1, le=100)
    least_similar: bool = False
    embedding_set: str = Field(default="default")
    provider: Literal["openai", "sentence-transformers"] = Field(default="openai")
    model: str = Field(default="text-embedding-3-large")
    dimensions: int | None = None
    device: str | None = None
    instruction: str | None = None
    include_analysis: bool = False
    include_plot: bool = False
    analysis_preview_chars: int = Field(default=480, ge=50, le=2000)
    plot_preview_chars: int = Field(default=480, ge=50, le=2000)


class TextClientRequest(BaseModel):
    provider: str = Field(default="sentence-transformers")
    model: str
    device: str | None = None
    instruction: str | None = None


class TitleSearchResponse(BaseModel):
    results: list[TitleSummary]


class TitleDetailResponse(BaseModel):
    title: TitleSummary
    plot: str | None = None
    analysis: str | None = None
    suggestions: list[TitleSummary] | None = None


class RecommendationResponse(BaseModel):
    seeds: list[TitleSummary]
    results: list[Recommendation]


class TextMatch(BaseModel):
    tconst: str
    primary_title: str | None
    start_year: int | None
    snippet: str


class TextSearchResponse(BaseModel):
    matches: list[TextMatch]


class ExplanationRequest(BaseModel):
    profile: str = Field(..., min_length=1)
    seed_ids: list[str] = Field(min_items=1, max_items=10)
    candidate_id: str
    candidate_score: float | None = None


class ExplanationResponse(BaseModel):
    explanation: str


# -----------------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------------


app = FastAPI(title="Movie Pipeline API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/profiles")
def list_profiles() -> dict[str, list[str]]:
    return {"profiles": _discover_profiles()}


@app.get("/api/indexes")
def list_indexes() -> dict[str, Sequence[str]]:
    return {"indexes": EMBEDDING_INDEX_CHOICES}


@app.get("/api/embedding-sets", response_model=EmbeddingSetResponse)
def list_embedding_sets(profile: str = Query(..., min_length=1)) -> EmbeddingSetResponse:
    manifest = _manifest(profile)
    entries = manifest.list_embedding_variants(profile=profile)
    variants = [
        EmbeddingSetInfo(
            name=str(entry.get("name") or "").strip(),
            provider=str(entry.get("provider")) if entry.get("provider") else "openai",
            model=entry.get("model"),
            dimension=entry.get("dimension"),
            vector_count=int(entry.get("vector_count", 0) or 0),
            updated_at=entry.get("updated_at"),
        )
        for entry in entries
        if str(entry.get("name") or "").strip()
    ]
    variants.sort(key=lambda item: item.name)
    return EmbeddingSetResponse(embedding_sets=variants)


@app.get("/api/manifest/summary", response_model=ManifestSummaryResponse)
def manifest_summary(
    profile: str = Query(..., min_length=1),
    variant: str | None = Query(None, alias="embedding_set"),
) -> ManifestSummaryResponse:
    manifest = _manifest(profile)
    stats = manifest.summary_stats(profile=profile, variant=variant)
    return ManifestSummaryResponse(**stats.__dict__)

@app.get("/api/titles/search", response_model=TitleSearchResponse)
def search_titles(
    q: str = Query(min_length=1),
    profile: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
) -> TitleSearchResponse:
    manifest = _manifest(profile)
    matches = manifest.search_titles(q, limit=limit)
    return TitleSearchResponse(results=[TitleSummary(**_title_to_payload(record)) for record in matches])


@app.get("/api/titles/{identifier}", response_model=TitleDetailResponse)
def fetch_title(
    identifier: str,
    profile: str = Query(..., min_length=1),
    matches: int = Query(5, ge=1, le=20),
    include_plot: bool = Query(False),
    include_analysis: bool = Query(False),
) -> TitleDetailResponse:
    manifest = _manifest(profile)
    record: TitleRecord | None = None
    suggestions: list[TitleSummary] | None = None
    if identifier.lower().startswith("tt"):
        record = manifest.get_title(identifier)
        if not record:
            raise HTTPException(status_code=404, detail=f"No title found for {identifier}.")
    else:
        results = manifest.search_titles(identifier, limit=matches)
        if not results:
            raise HTTPException(status_code=404, detail=f"No matches for '{identifier}'.")
        record = results[0]
        if len(results) > 1:
            suggestions = [
                TitleSummary(**_title_to_payload(candidate)) for candidate in results[1:]
            ]
    payload = _title_to_payload(record)
    plot_text = _read_text(_plot_path(record.tconst)) if include_plot else None
    analysis_text = _read_text(_analysis_path(profile, record.tconst)) if include_analysis else None
    return TitleDetailResponse(
        title=TitleSummary(**payload),
        plot=plot_text,
        analysis=analysis_text,
        suggestions=suggestions,
    )


@app.post("/api/query/title", response_model=RecommendationResponse)
def query_by_title(request: TitleQueryRequest) -> RecommendationResponse:
    profile = _normalize_profile(request.profile)
    manifest = _manifest(profile)
    seed_records = _resolve_titles(manifest, request.identifiers, matches=request.matches)
    embedding_entry = _load_embeddings(
        request.index,
        profile,
        request.embedding_set,
    )

    vectors = []
    for record in seed_records:
        idx = embedding_entry.index.get(record.tconst)
        if idx is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No embedding stored for {record.tconst}. "
                    "Re-run the embedding pipeline for this profile/variant."
                ),
            )
        vector = embedding_entry.matrix[idx]
        vector_dim = vector.shape[-1]
        if vector_dim != embedding_entry.dimension:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Vector dimension mismatch for {record.tconst}: "
                    f"{vector_dim} vs matrix dim {embedding_entry.dimension}"
                ),
            )
        vectors.append(vector)

    if request.score_mode == "centroid":
        if request.weights:
            if len(request.weights) != len(seed_records):
                raise HTTPException(
                    status_code=400,
                    detail="Number of weights must match number of seeds.",
                )
            if any(weight < 0 for weight in request.weights):
                raise HTTPException(status_code=400, detail="Weights must be non-negative.")
            total = sum(request.weights)
            if not 0.999 <= total <= 1.001:
                raise HTTPException(status_code=400, detail="Weights must sum to 1.0.")
        query_vector = combine_embeddings(vectors, weights=request.weights)
        results = top_k_similar(
            embedding_entry.matrix,
            query_vector,
            embedding_entry.ids,
            top_k=request.top_k,
            largest=not request.least_similar,
        )
    else:
        if request.weights:
            raise HTTPException(
                status_code=400,
                detail="weights can only be used with score_mode='centroid'.",
            )
        results = intersection_similar(
            embedding_entry.matrix,
            vectors,
            embedding_entry.ids,
            top_k=request.top_k,
            largest=not request.least_similar,
        )

    recommendations = _build_recommendations(
        manifest,
        profile=profile,
        matches=results,
        include_analysis=request.include_analysis,
        include_plot=request.include_plot,
        analysis_chars=request.analysis_preview_chars,
        plot_chars=request.plot_preview_chars,
    )
    return RecommendationResponse(
        seeds=[TitleSummary(**_title_to_payload(record)) for record in seed_records],
        results=recommendations,
    )


@app.post("/api/text/client/hold")
def hold_text_client(request: TextClientRequest) -> dict[str, object]:
    provider = _normalize_provider(request.provider, default="sentence-transformers")
    if provider not in LOCAL_EMBEDDING_PROVIDERS:
        return {"status": "ignored", "provider": provider}
    instruction = resolve_embedding_instruction(provider, request.model, request.instruction, for_query=True)
    config = EmbeddingConfig(
        provider=provider,
        model=request.model,
        device=request.device,
        instruction=instruction,
        dimensions=None,
        batch_size=1,
    )
    client = _LOCAL_MODEL_REGISTRY.get(config)
    if client is None:
        try:
            client = _create_local_embedding_client(config)
        except RuntimeError as exc:  # pragma: no cover - depends on model availability
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    _LOCAL_MODEL_REGISTRY.ensure(config, client)
    return {"status": "ready", "provider": provider, "model": request.model}


@app.post("/api/text/client/release")
def release_text_client(request: TextClientRequest) -> dict[str, object]:
    provider = _normalize_provider(request.provider, default="sentence-transformers")
    if provider not in LOCAL_EMBEDDING_PROVIDERS:
        return {"status": "ignored", "provider": provider}
    instruction = resolve_embedding_instruction(provider, request.model, request.instruction, for_query=True)
    config = EmbeddingConfig(
        provider=provider,
        model=request.model,
        device=request.device,
        instruction=instruction,
        dimensions=None,
        batch_size=1,
    )
    outcome = _LOCAL_MODEL_REGISTRY.release(config)
    if outcome is None:
        return {"status": "missing", "provider": provider, "model": request.model}
    if outcome:
        return {"status": "released", "provider": provider, "model": request.model}
    return {"status": "retained", "provider": provider, "model": request.model}


@app.post("/api/query/text", response_model=RecommendationResponse)
def query_by_text(request: TextQueryRequest) -> RecommendationResponse:
    profile = _normalize_profile(request.profile)
    embedding_entry = _load_embeddings(
        request.index,
        profile,
        request.embedding_set,
    )
    matrix = embedding_entry.matrix
    stored_dim = embedding_entry.dimension
    target_dim = request.dimensions or stored_dim
    provider = _normalize_provider(request.provider, default="openai")
    instruction = resolve_embedding_instruction(
        provider,
        request.model,
        request.instruction,
        for_query=True,
    )
    config = EmbeddingConfig(
        provider=provider,
        model=request.model,
        embedding_set=request.embedding_set,
        dimensions=target_dim if provider == "openai" else None,
        device=request.device,
        instruction=instruction,
        batch_size=1,
    )
    try:
        with _embedding_client_session(config) as client:
            embeddings, _ = client.embed([request.text])
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    query_vector = embeddings[0]
    if len(query_vector) != stored_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Query embedding dimension mismatch: expected {stored_dim}, "
                f"received {len(query_vector)}"
            ),
        )
    results = top_k_similar(
        matrix,
        query_vector,
        embedding_entry.ids,
        top_k=request.top_k,
        largest=not request.least_similar,
    )
    manifest = _manifest(profile)
    recommendations = _build_recommendations(
        manifest,
        profile=profile,
        matches=results,
        include_analysis=request.include_analysis,
        include_plot=request.include_plot,
        analysis_chars=request.analysis_preview_chars,
        plot_chars=request.plot_preview_chars,
    )
    seed_summary = TitleSummary(
        tconst="free-text",
        primary_title=f"Query: {request.text[:80]}",
        original_title=None,
        title_type="query",
        start_year=None,
        end_year=None,
        runtime_minutes=None,
        genres=None,
        num_votes=0,
        average_rating=0.0,
        sort_rank=0,
    )
    return RecommendationResponse(seeds=[seed_summary], results=recommendations)


def _build_openai_client(config: EmbeddingConfig) -> OpenAIEmbeddingClient:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for text queries.")
    return OpenAIEmbeddingClient(api_key, config)


def _create_local_embedding_client(
    config: EmbeddingConfig,
) -> SentenceTransformerEmbeddingClient | QwenEmbeddingClient:
    if is_qwen_embedding_model(config.model):
        return QwenEmbeddingClient(config)
    return SentenceTransformerEmbeddingClient(config)


def _build_recommendations(
    manifest: Manifest,
    *,
    profile: str,
    matches: Iterable[tuple[str, float]],
    include_analysis: bool,
    include_plot: bool,
    analysis_chars: int,
    plot_chars: int,
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []
    for imdb_id, score in matches:
        record = manifest.get_title(imdb_id)
        if not record:
            continue
        analysis_preview = None
        plot_preview = None
        if include_analysis:
            analysis_preview = _truncate_text(
                _read_text(_analysis_path(profile, imdb_id)),
                max_chars=analysis_chars,
            )
        if include_plot:
            plot_preview = _truncate_text(
                _read_text(_plot_path(imdb_id)),
                max_chars=plot_chars,
            )
        recommendations.append(
            Recommendation(
                title=TitleSummary(**_title_to_payload(record)),
                score=score,
                analysis_preview=analysis_preview,
                plot_preview=plot_preview,
            )
        )
    return recommendations


@app.get("/api/text/search", response_model=TextSearchResponse)
def search_text(
    target: Literal["analysis", "plot"] = Query(...),
    profile: str = Query(..., min_length=1),
    q: str = Query(min_length=2),
    limit: int = Query(10, ge=1, le=50),
) -> TextSearchResponse:
    matches = _text_search_matches(
        target=target,
        profile=profile,
        query=q,
        limit=limit,
    )
    return TextSearchResponse(matches=matches)


@app.post("/api/explain", response_model=ExplanationResponse)
def explain_recommendation(request: ExplanationRequest) -> ExplanationResponse:
    profile = _normalize_profile(request.profile)
    manifest = _manifest(profile)
    seed_records = _resolve_titles(manifest, request.seed_ids, matches=5)
    if not seed_records:
        raise HTTPException(status_code=400, detail="No valid seeds supplied.")
    try:
        candidate_record = _resolve_titles(manifest, [request.candidate_id], matches=5)[0]
    except HTTPException as exc:
        raise exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summaries: list[str] = []
    labels: list[str] = []
    for record in seed_records:
        labels.append(_format_label(record))
        summaries.append(f"{labels[-1]}\n{_require_analysis(profile, record.tconst)}")
    seed_label = (
        f"Combined seeds: {' + '.join(labels)}" if len(labels) > 1 else labels[0]
    )
    seed_context = SeedContext(label=seed_label, summary="\n\n".join(summaries))

    candidate_text = _require_analysis(profile, candidate_record.tconst)
    candidate_label = _format_label(candidate_record)
    candidate_context = CandidateContext(
        label=candidate_label,
        score=request.candidate_score or 0.0,
        summary=candidate_text,
    )
    try:
        explanation = _get_explainer().explain(seed_context, [candidate_context])
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Explanation generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate explanation.") from exc
    return ExplanationResponse(explanation=explanation)


# Entry point for `uvicorn movie_pipeline.server:app`
__all__ = ["app"]

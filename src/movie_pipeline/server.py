"""FastAPI server that exposes the movie pipeline features over HTTP."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Literal, Sequence

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .embeddings import (
    EmbeddingConfig,
    OpenAIEmbeddingClient,
    combine_embeddings,
    intersection_similar,
    load_embedding_vector,
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_profile(profile: str | None) -> str:
    return (profile or "default").strip() or "default"


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
    profiles = {"default"}
    for base in (PATHS.analyses, PATHS.embeddings):
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir():
                profiles.add(child.name)
    return sorted(profiles)


@lru_cache(maxsize=32)
def _load_embeddings(index: str, profile: str):
    if index == "analysis":
        directory = embeddings_dir(profile)
    elif index == "plot":
        PATHS.plot_embeddings.mkdir(parents=True, exist_ok=True)
        directory = PATHS.plot_embeddings
    else:
        raise HTTPException(status_code=400, detail=f"Unknown embedding index '{index}'.")
    ids, matrix, dims = load_embeddings_matrix(directory)
    if matrix.size == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No embeddings found for index '{index}' (profile '{profile}').",
        )
    return ids, matrix, dims, directory


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
        results = manifest.search_titles(clean, limit=matches)
        if not results:
            raise HTTPException(status_code=404, detail=f"No titles matched '{clean}'.")
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


class TitleQueryRequest(BaseModel):
    profile: str = Field(default="default")
    index: Literal["analysis", "plot"] = Field(default="analysis")
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
    profile: str = Field(default="default")
    index: Literal["analysis", "plot"] = Field(default="analysis")
    text: str = Field(min_length=3, max_length=5000)
    top_k: int = Field(default=10, ge=1, le=100)
    model: str = Field(default="text-embedding-3-large")
    dimensions: int | None = None
    include_analysis: bool = False
    include_plot: bool = False
    analysis_preview_chars: int = Field(default=480, ge=50, le=2000)
    plot_preview_chars: int = Field(default=480, ge=50, le=2000)


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
    profile: str = Field(default="default")
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


@app.get("/api/titles/search", response_model=TitleSearchResponse)
def search_titles(
    q: str = Query(min_length=1),
    profile: str = Query("default"),
    limit: int = Query(10, ge=1, le=50),
) -> TitleSearchResponse:
    manifest = _manifest(profile)
    matches = manifest.search_titles(q, limit=limit)
    return TitleSearchResponse(results=[TitleSummary(**_title_to_payload(record)) for record in matches])


@app.get("/api/titles/{identifier}", response_model=TitleDetailResponse)
def fetch_title(
    identifier: str,
    profile: str = Query("default"),
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
    ids, matrix, stored_dim, _directory = _load_embeddings(request.index, profile)

    vectors = []
    for record in seed_records:
        vector, vector_dim = load_embedding_vector(
            record.tconst,
            vectors_dir=_directory,
        )
        if vector_dim != stored_dim:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Vector dimension mismatch for {record.tconst}: "
                    f"{vector_dim} vs matrix dim {stored_dim}"
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
            matrix,
            query_vector,
            ids,
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
            matrix,
            vectors,
            ids,
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


@app.post("/api/query/text", response_model=RecommendationResponse)
def query_by_text(request: TextQueryRequest) -> RecommendationResponse:
    profile = _normalize_profile(request.profile)
    ids, matrix, stored_dim, _ = _load_embeddings(request.index, profile)
    target_dim = request.dimensions or stored_dim
    config = EmbeddingConfig(model=request.model, dimensions=target_dim)
    try:
        client = _build_embedding_client(config)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        embeddings, _ = client.embed([request.text])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    query_vector = embeddings[0]
    if len(query_vector) != stored_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Query vector dimension {len(query_vector)} "
                f"does not match stored dim {stored_dim}."
            ),
        )
    results = top_k_similar(
        matrix,
        query_vector,
        ids,
        top_k=request.top_k,
        largest=True,
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


def _build_embedding_client(config: EmbeddingConfig) -> OpenAIEmbeddingClient:
    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for text queries.")
    return OpenAIEmbeddingClient(api_key, config)


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
    profile: str = Query("default"),
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

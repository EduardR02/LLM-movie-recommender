"""SQLite-backed manifest for pipeline artifacts."""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable, Iterator, Literal, Sequence

from .paths import PATHS

PlotStatus = Literal["queued", "ok", "missing", "error"]
AnalysisStatus = Literal["queued", "running", "ok", "error", "needs_retry"]
EmbeddingStatus = Literal["queued", "ok", "error"]


@dataclass
class TitleRecord:
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


@dataclass
class AnalysisCandidate:
    title: TitleRecord
    plot_source: str | None
    plot_path: Path
    plot_hash: str | None
    analysis_status: str | None
    attempts: int


@dataclass
class EmbeddingCandidate:
    title: TitleRecord
    analysis_path: Path
    analysis_status: str | None
    embedding_status: str | None
    variant: str


@dataclass
class SessionRecord:
    id: str
    component: str
    started_at: str
    completed_at: str | None
    total_input_tokens: int
    total_cached_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int
    total_cost: float
    notes: str | None


@dataclass
class PlotRecord:
    status: str
    source: str | None
    raw_path: str | None
    clean_path: str | None
    error: str | None
    plot_hash: str | None


@dataclass
class ManifestSummaryStats:
    total_titles: int
    plot_count: int
    wikipedia_articles: int
    analysis_count: int
    embedding_count: int


def _normalize_profile(value: str | None) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError("Profile value is required.")
    return normalized


def _normalize_variant(value: str | None) -> str:
    return (value or "default").strip() or "default"


def _row_to_title(row: sqlite3.Row) -> TitleRecord:
    return TitleRecord(
        tconst=row["tconst"],
        primary_title=row["primary_title"],
        original_title=row["original_title"],
        title_type=row["title_type"],
        start_year=row["start_year"],
        end_year=row["end_year"],
        runtime_minutes=row["runtime_minutes"],
        genres=row["genres"],
        num_votes=row["num_votes"],
        average_rating=row["average_rating"],
        sort_rank=row["sort_rank"],
    )


class Manifest:
    """Manage pipeline metadata and persistence."""

    def __init__(self, path: Path | None = None, *, profile: str | None = None) -> None:
        self.profile = profile.strip() if isinstance(profile, str) else None
        self.path = path or (PATHS.state / "manifest.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    # ---------------------------------------------------------------------
    # Schema management
    # ---------------------------------------------------------------------

    def _initialize_schema(self) -> None:
        with self.cursor() as cur:
            for statement in _CREATE_TABLE_STATEMENTS:
                cur.execute(statement)
            for statement in _CREATE_INDEX_STATEMENTS:
                cur.execute(statement)
            for statement in _TRIGGER_STATEMENTS:
                cur.execute(statement)

    # ---------------------------------------------------------------------
    # Title management
    # ---------------------------------------------------------------------

    def reset_titles(self) -> None:
        with self.cursor() as cur:
            cur.execute("DELETE FROM titles;")

    def bulk_insert_titles(self, records: Iterable[TitleRecord]) -> None:
        payload: list[tuple] = [
            (
                r.tconst,
                r.primary_title,
                r.original_title,
                r.title_type,
                r.start_year,
                r.end_year,
                r.runtime_minutes,
                r.genres,
                r.num_votes,
                r.average_rating,
                r.sort_rank,
            )
            for r in records
        ]
        if not payload:
            return
        with self.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO titles (
                    tconst,
                    primary_title,
                    original_title,
                    title_type,
                    start_year,
                    end_year,
                    runtime_minutes,
                    genres,
                    num_votes,
                    average_rating,
                    sort_rank
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tconst) DO UPDATE SET
                    primary_title=excluded.primary_title,
                    original_title=excluded.original_title,
                    title_type=excluded.title_type,
                    start_year=excluded.start_year,
                    end_year=excluded.end_year,
                    runtime_minutes=excluded.runtime_minutes,
                    genres=excluded.genres,
                    num_votes=excluded.num_votes,
                    average_rating=excluded.average_rating,
                    sort_rank=excluded.sort_rank;
                """,
                payload,
            )

    def iter_titles(self) -> Iterator[TitleRecord]:
        query = """
            SELECT
                tconst,
                primary_title,
                original_title,
                title_type,
                start_year,
                end_year,
                runtime_minutes,
                genres,
                num_votes,
                average_rating,
                sort_rank
            FROM titles
            ORDER BY sort_rank ASC;
        """
        with self.cursor() as cur:
            for row in cur.execute(query):
                yield _row_to_title(row)

    def iter_latest_titles(self, limit: int = 10) -> Iterator[TitleRecord]:
        query = """
            SELECT
                tconst,
                primary_title,
                original_title,
                title_type,
                start_year,
                end_year,
                runtime_minutes,
                genres,
                num_votes,
                average_rating,
                sort_rank
            FROM titles
            WHERE start_year IS NOT NULL
            ORDER BY start_year DESC, num_votes DESC, average_rating DESC
            LIMIT ?;
        """
        with self.cursor() as cur:
            for row in cur.execute(query, (limit,)):
                yield _row_to_title(row)

    def get_title(self, tconst: str) -> TitleRecord | None:
        query = """
            SELECT
                tconst,
                primary_title,
                original_title,
                title_type,
                start_year,
                end_year,
                runtime_minutes,
                genres,
                num_votes,
                average_rating,
                sort_rank
            FROM titles
            WHERE tconst = ?;
        """
        with self.cursor() as cur:
            row = cur.execute(query, (tconst,)).fetchone()
        return _row_to_title(row) if row else None

    def search_titles(self, term: str, limit: int = 5) -> list[TitleRecord]:
        normalized = term.strip()
        if not normalized:
            return []
        target_year: int | None = None
        title_only = normalized
        year_match = TITLE_WITH_YEAR_RE.match(normalized)
        if year_match:
            title_only = year_match.group("title").strip() or normalized
            try:
                target_year = int(year_match.group("year"))
            except ValueError:
                target_year = None
        like_pattern = f"%{title_only}%"
        query = """
            SELECT
                tconst,
                primary_title,
                original_title,
                title_type,
                start_year,
                end_year,
                runtime_minutes,
                genres,
                num_votes,
                average_rating,
                sort_rank
            FROM titles
            WHERE primary_title LIKE ?
            ORDER BY
                CASE LOWER(primary_title) WHEN LOWER(?) THEN 0 ELSE 1 END,
                CASE WHEN ? IS NOT NULL AND start_year = ? THEN 0 ELSE 1 END,
                sort_rank ASC
            LIMIT ?;
        """
        with self.cursor() as cur:
            rows = cur.execute(
                query,
                (
                    like_pattern,
                    title_only,
                    target_year,
                    target_year,
                    limit,
                ),
            ).fetchall()
        return [_row_to_title(row) for row in rows]

    # ---------------------------------------------------------------------
    # Plot management
    # ---------------------------------------------------------------------

    def update_plot_status(
        self,
        tconst: str,
        status: PlotStatus,
        *,
        source: str | None = None,
        content_hash: str | None = None,
        raw_path: str | None = None,
        clean_path: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO plots (
                    tconst,
                    status,
                    source,
                    content_hash,
                    raw_path,
                    clean_path,
                    error,
                    fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(tconst) DO UPDATE SET
                    status=excluded.status,
                    source=excluded.source,
                    content_hash=excluded.content_hash,
                    raw_path=excluded.raw_path,
                    clean_path=excluded.clean_path,
                    error=excluded.error,
                    fetched_at=excluded.fetched_at;
                """,
                (tconst, status, source, content_hash, raw_path, clean_path, error),
            )

    def get_plot_record(self, tconst: str) -> PlotRecord | None:
        query = """
            SELECT status, source, raw_path, clean_path, error, content_hash
            FROM plots
            WHERE tconst = ?;
        """
        with self.cursor() as cur:
            row = cur.execute(query, (tconst,)).fetchone()
        if not row:
            return None
        return PlotRecord(
            status=row["status"],
            source=row["source"],
            raw_path=row["raw_path"],
            clean_path=row["clean_path"],
            error=row["error"],
            plot_hash=row["content_hash"],
        )

    # ---------------------------------------------------------------------
    # Analysis tracking
    # ---------------------------------------------------------------------

    def update_analysis_status(
        self,
        tconst: str,
        status: AnalysisStatus,
        *,
        profile: str | None = None,
        session_id: str | None = None,
        model: str | None = None,
        system_prompt_hash: str | None = None,
        plot_hash: str | None = None,
        attempts: int | None = None,
        input_tokens: int | None = None,
        input_cached_tokens: int | None = None,
        output_tokens: int | None = None,
        output_reasoning_tokens: int | None = None,
        cost_estimate: float | None = None,
        output_path: str | None = None,
        error: str | None = None,
    ) -> None:
        active_profile = _normalize_profile(profile or self.profile)
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analyses (
                    tconst,
                    profile,
                    status,
                    session_id,
                    model,
                    system_prompt_hash,
                    plot_hash,
                    attempts,
                    input_tokens,
                    input_cached_tokens,
                    output_tokens,
                    output_reasoning_tokens,
                    cost_estimate,
                    output_path,
                    error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tconst, profile) DO UPDATE SET
                    status=excluded.status,
                    session_id=COALESCE(excluded.session_id, analyses.session_id),
                    model=COALESCE(excluded.model, analyses.model),
                    system_prompt_hash=COALESCE(excluded.system_prompt_hash, analyses.system_prompt_hash),
                    plot_hash=COALESCE(excluded.plot_hash, analyses.plot_hash),
                    attempts=COALESCE(excluded.attempts, analyses.attempts),
                    input_tokens=COALESCE(excluded.input_tokens, analyses.input_tokens),
                    input_cached_tokens=COALESCE(excluded.input_cached_tokens, analyses.input_cached_tokens),
                    output_tokens=COALESCE(excluded.output_tokens, analyses.output_tokens),
                    output_reasoning_tokens=COALESCE(
                        excluded.output_reasoning_tokens, analyses.output_reasoning_tokens
                    ),
                    cost_estimate=COALESCE(excluded.cost_estimate, analyses.cost_estimate),
                    output_path=COALESCE(excluded.output_path, analyses.output_path),
                    error=excluded.error;
                """,
                (
                    tconst,
                    active_profile,
                    status,
                    session_id,
                    model,
                    system_prompt_hash,
                    plot_hash,
                    attempts,
                    input_tokens,
                    input_cached_tokens,
                    output_tokens,
                    output_reasoning_tokens,
                    cost_estimate,
                    output_path,
                    error,
                ),
            )

    def ensure_analysis_record(
        self,
        record: TitleRecord,
        *,
        profile: str,
        plot_hash: str | None,
        path: Path,
        model: str,
    ) -> None:
        active_profile = _normalize_profile(profile)
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analyses (
                    tconst,
                    profile,
                    status,
                    session_id,
                    model,
                    system_prompt_hash,
                    plot_hash,
                    attempts,
                    input_tokens,
                    output_tokens,
                    cost_estimate,
                    output_path,
                    error
                ) VALUES (?, ?, 'ok', ?, ?, NULL, ?, 1, NULL, NULL, NULL, ?, NULL)
                ON CONFLICT(tconst, profile) DO UPDATE SET
                    status='ok',
                    model=excluded.model,
                    plot_hash=COALESCE(excluded.plot_hash, analyses.plot_hash),
                    output_path=excluded.output_path,
                    error=NULL,
                    attempts=COALESCE(analyses.attempts, 0)+1;
                """,
                (
                    record.tconst,
                    active_profile,
                    "manual-import",
                    model,
                    plot_hash,
                    str(path),
                ),
            )

    def iter_analysis_tconsts(
        self,
        statuses: Sequence[str] | None = None,
        *,
        profile: str | None = None,
    ) -> Iterator[str]:
        active_profile = _normalize_profile(profile or self.profile)
        query = "SELECT tconst FROM analyses WHERE profile = ?"
        params: list[object] = [active_profile]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query = f"{query} AND status IN ({placeholders})"
            params.extend(statuses)
        with self.cursor() as cur:
            for row in cur.execute(query, params):
                yield row["tconst"]

    def reset_analysis(self, tconst: str, *, profile: str | None = None) -> None:
        active_profile = _normalize_profile(profile or self.profile)
        with self.cursor() as cur:
            plot_row = cur.execute(
                "SELECT content_hash FROM plots WHERE tconst = ?",
                (tconst,),
            ).fetchone()
            plot_hash = plot_row["content_hash"] if plot_row else None
            exists = cur.execute(
                "SELECT 1 FROM analyses WHERE tconst = ? AND profile = ?",
                (tconst, active_profile),
            ).fetchone()
            if exists:
                cur.execute(
                    """
                    UPDATE analyses
                    SET status='queued',
                        session_id=NULL,
                        model=NULL,
                        system_prompt_hash=NULL,
                        plot_hash=?,
                        attempts=0,
                        input_tokens=NULL,
                        input_cached_tokens=NULL,
                        output_tokens=NULL,
                        output_reasoning_tokens=NULL,
                        cost_estimate=NULL,
                        output_path=NULL,
                        error=NULL
                    WHERE tconst=? AND profile=?;
                    """,
                    (plot_hash, tconst, active_profile),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO analyses (
                        tconst,
                        profile,
                        status,
                        session_id,
                        model,
                        system_prompt_hash,
                        plot_hash,
                        attempts,
                        input_tokens,
                        input_cached_tokens,
                        output_tokens,
                        output_reasoning_tokens,
                        cost_estimate,
                        output_path,
                        error
                    ) VALUES (?, ?, 'queued', NULL, NULL, NULL, ?, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
                    """,
                    (tconst, active_profile, plot_hash),
                )

    def iter_analysis_candidates(
        self,
        limit: int | None = None,
        *,
        include_completed: bool = False,
        profile: str | None = None,
    ) -> Iterator[AnalysisCandidate]:
        active_profile = _normalize_profile(profile or self.profile)
        query = """
            SELECT
                t.tconst,
                t.primary_title,
                t.original_title,
                t.title_type,
                t.start_year,
                t.end_year,
                t.runtime_minutes,
                t.genres,
                t.num_votes,
                t.average_rating,
                t.sort_rank,
                p.source,
                p.clean_path,
                p.content_hash,
                COALESCE(a.status, '') AS analysis_status,
                COALESCE(a.attempts, 0) AS attempts
            FROM titles AS t
            JOIN plots AS p ON p.tconst = t.tconst AND p.status = 'ok'
            LEFT JOIN analyses AS a ON a.tconst = t.tconst AND a.profile = ?
        """
        params: list[object] = [active_profile]
        conditions: list[str] = []
        if not include_completed:
            conditions.append("COALESCE(a.status, '') != 'ok'")
        if conditions:
            query = f"{query} WHERE {' AND '.join(conditions)}"
        query = f"{query} ORDER BY t.sort_rank ASC"
        if limit is not None:
            query = f"{query} LIMIT ?"
            params.append(int(limit))

        with self.cursor() as cur:
            for row in cur.execute(query, params):
                title = _row_to_title(row)
                yield AnalysisCandidate(
                    title=title,
                    plot_source=row["source"],
                    plot_path=Path(row["clean_path"]),
                    plot_hash=row["content_hash"],
                    analysis_status=row["analysis_status"] or None,
                    attempts=row["attempts"],
                )

    # ---------------------------------------------------------------------
    # Embedding tracking
    # ---------------------------------------------------------------------

    def update_embedding_status(
        self,
        tconst: str,
        status: EmbeddingStatus,
        *,
        profile: str | None = None,
        variant: str | None = None,
        session_id: str | None = None,
        model: str | None = None,
        prompt_hash: str | None = None,
        dim: int | None = None,
        input_tokens: int | None = None,
        vector_path: str | None = None,
        provider: str | None = None,
        error: str | None = None,
    ) -> None:
        active_profile = _normalize_profile(profile or self.profile)
        active_variant = _normalize_variant(variant)
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddings (
                    tconst,
                    profile,
                    variant,
                    provider,
                    status,
                    session_id,
                    model,
                    prompt_hash,
                    dim,
                    input_tokens,
                    vector_path,
                    error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tconst, profile, variant) DO UPDATE SET
                    status=excluded.status,
                    provider=COALESCE(NULLIF(excluded.provider, ''), embeddings.provider),
                    session_id=COALESCE(excluded.session_id, embeddings.session_id),
                    model=COALESCE(excluded.model, embeddings.model),
                    prompt_hash=COALESCE(excluded.prompt_hash, embeddings.prompt_hash),
                    dim=COALESCE(excluded.dim, embeddings.dim),
                    input_tokens=COALESCE(excluded.input_tokens, embeddings.input_tokens),
                    vector_path=COALESCE(excluded.vector_path, embeddings.vector_path),
                    error=excluded.error;
                """,
                (
                    tconst,
                    active_profile,
                    active_variant,
                    provider or "openai",
                    status,
                    session_id,
                    model,
                    prompt_hash,
                    dim,
                    input_tokens,
                    vector_path,
                    error,
                ),
            )

    def iter_embedding_candidates(
        self,
        limit: int | None = None,
        *,
        include_completed: bool = False,
        profile: str | None = None,
        variant: str | None = None,
    ) -> Iterator[EmbeddingCandidate]:
        active_profile = _normalize_profile(profile or self.profile)
        active_variant = _normalize_variant(variant)
        query = """
            SELECT
                t.tconst,
                t.primary_title,
                t.original_title,
                t.title_type,
                t.start_year,
                t.end_year,
                t.runtime_minutes,
                t.genres,
                t.num_votes,
                t.average_rating,
                t.sort_rank,
                a.output_path,
                a.status AS analysis_status,
                COALESCE(e.status, '') AS embedding_status,
                COALESCE(e.variant, ?) AS resolved_variant
            FROM titles AS t
            JOIN analyses AS a ON a.tconst = t.tconst AND a.profile = ? AND a.status = 'ok'
            LEFT JOIN embeddings AS e
                ON e.tconst = t.tconst AND e.profile = ? AND e.variant = ?
        """
        params: list[object] = [active_variant, active_profile, active_profile, active_variant]
        conditions: list[str] = []
        if not include_completed:
            conditions.append("COALESCE(e.status, '') != 'ok'")
        if conditions:
            query = f"{query} WHERE {' AND '.join(conditions)}"
        query = f"{query} ORDER BY t.sort_rank ASC"
        if limit is not None:
            query = f"{query} LIMIT ?"
            params.append(int(limit))

        with self.cursor() as cur:
            for row in cur.execute(query, params):
                analysis_path = row["output_path"]
                if not analysis_path:
                    continue
                title = _row_to_title(row)
                yield EmbeddingCandidate(
                    title=title,
                    analysis_path=Path(analysis_path),
                    analysis_status=row["analysis_status"] or None,
                    embedding_status=row["embedding_status"] or None,
                    variant=row["resolved_variant"],
                )

    def list_embedding_variants(self, *, profile: str | None = None) -> list[dict[str, object]]:
        active_profile = _normalize_profile(profile or self.profile)
        query = """
            SELECT
                variant,
                MAX(COALESCE(NULLIF(provider, ''), 'openai')) AS provider,
                MAX(NULLIF(model, '')) AS model,
                MAX(COALESCE(dim, 0)) AS dim,
                COUNT(*) AS vector_count,
                MAX(updated_at) AS updated_at
            FROM embeddings
            WHERE profile = ? AND status = 'ok'
            GROUP BY variant;
        """
        variants: list[dict[str, object]] = []
        with self.cursor() as cur:
            for row in cur.execute(query, (active_profile,)):
                name = (row["variant"] or "").strip()
                if not name:
                    # Skip rows that do not specify a variant; callers should normalise these.
                    continue
                variants.append(
                    {
                        "name": name,
                        "provider": row["provider"] or "openai",
                        "model": row["model"],
                        "dimension": int(row["dim"] or 0) or None,
                        "vector_count": int(row["vector_count"] or 0),
                        "updated_at": row["updated_at"],
                    }
                )
        variants.sort(key=lambda item: item["name"])
        return variants

    # ---------------------------------------------------------------------
    # Sessions and reporting
    # ---------------------------------------------------------------------

    def register_session(self, session_id: str, component: str) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (id, component)
                VALUES (?, ?)
                ON CONFLICT(id) DO NOTHING;
                """,
                (session_id, component),
            )

    def finalize_session(
        self,
        session_id: str,
        *,
        total_input_tokens: int,
        total_cached_input_tokens: int,
        total_output_tokens: int,
        total_reasoning_tokens: int,
        total_cost: float,
        notes: str | None = None,
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                UPDATE sessions
                SET completed_at = CURRENT_TIMESTAMP,
                    total_input_tokens = ?,
                    total_cached_input_tokens = ?,
                    total_output_tokens = ?,
                    total_reasoning_tokens = ?,
                    total_cost = ?,
                    notes = ?
                WHERE id = ?;
                """,
                (
                    total_input_tokens,
                    total_cached_input_tokens,
                    total_output_tokens,
                    total_reasoning_tokens,
                    total_cost,
                    notes,
                    session_id,
                ),
            )

    def iter_sessions(self, limit: int = 10) -> Iterator[SessionRecord]:
        query = """
            SELECT
                id,
                component,
                started_at,
                completed_at,
                COALESCE(total_input_tokens, 0) AS total_input_tokens,
                COALESCE(total_cached_input_tokens, 0) AS total_cached_input_tokens,
                COALESCE(total_output_tokens, 0) AS total_output_tokens,
                COALESCE(total_reasoning_tokens, 0) AS total_reasoning_tokens,
                COALESCE(total_cost, 0.0) AS total_cost,
                notes
            FROM sessions
            ORDER BY started_at DESC
            LIMIT ?;
        """
        with self.cursor() as cur:
            for row in cur.execute(query, (limit,)):
                yield SessionRecord(
                    id=row["id"],
                    component=row["component"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    total_input_tokens=row["total_input_tokens"],
                    total_cached_input_tokens=row["total_cached_input_tokens"],
                    total_output_tokens=row["total_output_tokens"],
                    total_reasoning_tokens=row["total_reasoning_tokens"],
                    total_cost=row["total_cost"],
                    notes=row["notes"],
                )

    def session_totals(self) -> list[tuple[str, int, int, int, int, float]]:
        query = """
            SELECT
                component,
                COALESCE(SUM(total_input_tokens), 0) AS total_input_tokens,
                COALESCE(SUM(total_cached_input_tokens), 0) AS total_cached_input_tokens,
                COALESCE(SUM(total_output_tokens), 0) AS total_output_tokens,
                COALESCE(SUM(total_reasoning_tokens), 0) AS total_reasoning_tokens,
                COALESCE(SUM(total_cost), 0.0) AS total_cost
            FROM sessions
            GROUP BY component
            ORDER BY component ASC;
        """
        with self.cursor() as cur:
            return [tuple(row) for row in cur.execute(query)]

    def summary_stats(
        self,
        *,
        profile: str | None = None,
        variant: str | None = None,
    ) -> ManifestSummaryStats:
        active_profile = _normalize_profile(profile or self.profile)
        active_variant = _normalize_variant(variant)
        with self.cursor() as cur:
            total_titles = cur.execute("SELECT COUNT(*) AS count FROM titles;").fetchone()["count"]
            plots_row = cur.execute(
                """
                SELECT COUNT(*) AS ok_plots, COUNT(DISTINCT source) AS sources
                FROM plots
                WHERE status = 'ok';
                """
            ).fetchone()
            analyses_row = cur.execute(
                """
                SELECT COUNT(*) AS count
                FROM analyses
                WHERE profile = ? AND status = 'ok';
                """,
                (active_profile,),
            ).fetchone()
            embeddings_row = cur.execute(
                """
                SELECT COUNT(*) AS count
                FROM embeddings
                WHERE profile = ? AND variant = ? AND status = 'ok';
                """,
                (active_profile, active_variant),
            ).fetchone()
        return ManifestSummaryStats(
            total_titles=int(total_titles or 0),
            plot_count=int(plots_row["ok_plots"] or 0),
            wikipedia_articles=int(plots_row["sources"] or 0),
            analysis_count=int(analyses_row["count"] or 0),
            embedding_count=int(embeddings_row["count"] or 0),
        )


# -------------------------------------------------------------------------
# Schema definitions
# -------------------------------------------------------------------------

_CREATE_TABLE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS titles (
        tconst TEXT PRIMARY KEY,
        primary_title TEXT NOT NULL,
        original_title TEXT,
        title_type TEXT NOT NULL,
        start_year INTEGER,
        end_year INTEGER,
        runtime_minutes INTEGER,
        genres TEXT,
        num_votes INTEGER NOT NULL,
        average_rating REAL NOT NULL,
        sort_rank INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS plots (
        tconst TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'queued',
        source TEXT,
        content_hash TEXT,
        raw_path TEXT,
        clean_path TEXT,
        error TEXT,
        fetched_at TEXT,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (tconst) REFERENCES titles (tconst) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS analyses (
        tconst TEXT NOT NULL,
        profile TEXT NOT NULL DEFAULT 'default',
        status TEXT NOT NULL DEFAULT 'queued',
        session_id TEXT,
        model TEXT,
        system_prompt_hash TEXT,
        plot_hash TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        input_tokens INTEGER,
        input_cached_tokens INTEGER,
        output_tokens INTEGER,
        output_reasoning_tokens INTEGER,
        cost_estimate REAL,
        output_path TEXT,
        error TEXT,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (tconst, profile),
        FOREIGN KEY (tconst) REFERENCES titles (tconst) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        tconst TEXT NOT NULL,
        profile TEXT NOT NULL DEFAULT 'default',
        variant TEXT NOT NULL DEFAULT 'default',
        provider TEXT NOT NULL DEFAULT 'openai',
        status TEXT NOT NULL DEFAULT 'queued',
        session_id TEXT,
        model TEXT,
        prompt_hash TEXT,
        dim INTEGER,
        input_tokens INTEGER,
        vector_path TEXT,
        error TEXT,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (tconst, profile, variant),
        FOREIGN KEY (tconst) REFERENCES titles (tconst) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        component TEXT NOT NULL,
        started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        completed_at TEXT,
        total_input_tokens INTEGER NOT NULL DEFAULT 0,
        total_cached_input_tokens INTEGER NOT NULL DEFAULT 0,
        total_output_tokens INTEGER NOT NULL DEFAULT 0,
        total_reasoning_tokens INTEGER NOT NULL DEFAULT 0,
        total_cost REAL NOT NULL DEFAULT 0.0,
        notes TEXT
    );
    """,
)

_CREATE_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_plots_status ON plots(status);",
    "CREATE INDEX IF NOT EXISTS idx_analyses_profile_status ON analyses(profile, status);",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_profile_variant ON embeddings(profile, variant, status);",
)

_TRIGGER_STATEMENTS = (
    """
    CREATE TRIGGER IF NOT EXISTS trg_titles_updated
    AFTER UPDATE ON titles
    FOR EACH ROW
    BEGIN
        UPDATE titles
        SET updated_at = CURRENT_TIMESTAMP
        WHERE tconst = NEW.tconst;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_plots_updated
    AFTER UPDATE ON plots
    FOR EACH ROW
    BEGIN
        UPDATE plots
        SET updated_at = CURRENT_TIMESTAMP
        WHERE tconst = NEW.tconst;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_analyses_updated
    AFTER UPDATE ON analyses
    FOR EACH ROW
    BEGIN
        UPDATE analyses
        SET updated_at = CURRENT_TIMESTAMP
        WHERE tconst = NEW.tconst AND profile = NEW.profile;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_embeddings_updated
    AFTER UPDATE ON embeddings
    FOR EACH ROW
    BEGIN
        UPDATE embeddings
        SET updated_at = CURRENT_TIMESTAMP
        WHERE tconst = NEW.tconst AND profile = NEW.profile AND variant = NEW.variant;
    END;
    """,
)
TITLE_WITH_YEAR_RE = re.compile(r"^(?P<title>.+?)\s*\((?P<year>\d{4})\)$")

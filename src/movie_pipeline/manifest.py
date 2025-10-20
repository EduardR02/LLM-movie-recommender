"""SQLite-backed manifest for tracking pipeline artifacts."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

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
    plot_hash: str
    analysis_status: str | None
    attempts: int


@dataclass
class EmbeddingCandidate:
    title: TitleRecord
    analysis_path: Path
    analysis_status: str | None
    embedding_status: str | None


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


def _row_to_title(row: tuple) -> TitleRecord:
    return TitleRecord(
        tconst=row[0],
        primary_title=row[1],
        original_title=row[2],
        title_type=row[3],
        start_year=row[4],
        end_year=row[5],
        runtime_minutes=row[6],
        genres=row[7],
        num_votes=row[8],
        average_rating=row[9],
        sort_rank=row[10],
    )


class Manifest:
    """Manage the pipeline manifest database."""

    def __init__(self, path: Path | None = None, *, profile: str | None = None) -> None:
        normalized_profile = (profile or "default").strip()
        self.profile = normalized_profile or "default"
        self.path = path or (PATHS.state / "manifest.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._create_tables()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def _create_tables(self) -> None:
        with self.cursor() as cur:
            cur.execute(
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
                """
            )
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_titles_updated
                AFTER UPDATE ON titles
                FOR EACH ROW
                BEGIN
                    UPDATE titles
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE tconst = NEW.tconst;
                END;
                """
            )
            cur.execute(
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
                """
            )
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_plots_updated
                AFTER UPDATE ON plots
                FOR EACH ROW
                BEGIN
                    UPDATE plots
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE tconst = NEW.tconst;
                END;
                """
            )
            self._ensure_analysis_table(cur)
            self._ensure_embedding_table(cur)
            cur.execute(
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
                """
            )
            self._ensure_columns(
                cur,
                table="sessions",
                columns={
                    "total_cached_input_tokens": "INTEGER NOT NULL DEFAULT 0",
                    "total_reasoning_tokens": "INTEGER NOT NULL DEFAULT 0",
                },
            )

    def _ensure_analysis_table(self, cur: sqlite3.Cursor) -> None:
        cur.execute(ANALYSES_CREATE_SQL_IF_NOT_EXISTS)
        self._upgrade_table_to_profile(
            cur,
            table="analyses",
            create_sql=ANALYSES_CREATE_SQL,
            copy_columns=ANALYSES_BASE_COLUMNS,
        )
        cur.execute("DROP TRIGGER IF EXISTS trg_analyses_updated;")
        cur.execute(ANALYSES_TRIGGER_SQL)
        self._ensure_columns(
            cur,
            table="analyses",
            columns={
                "input_cached_tokens": "INTEGER",
                "output_reasoning_tokens": "INTEGER",
            },
        )

    def _ensure_embedding_table(self, cur: sqlite3.Cursor) -> None:
        cur.execute(EMBEDDINGS_CREATE_SQL_IF_NOT_EXISTS)
        self._upgrade_table_to_profile(
            cur,
            table="embeddings",
            create_sql=EMBEDDINGS_CREATE_SQL,
            copy_columns=EMBEDDINGS_BASE_COLUMNS,
        )
        cur.execute("DROP TRIGGER IF EXISTS trg_embeddings_updated;")
        cur.execute(EMBEDDINGS_TRIGGER_SQL)

    def _upgrade_table_to_profile(
        self,
        cur: sqlite3.Cursor,
        *,
        table: str,
        create_sql: str,
        copy_columns: list[str],
    ) -> None:
        info = cur.execute(f"PRAGMA table_info({table});").fetchall()
        if not info:
            cur.execute(create_sql)
            return
        column_names = {row[1] for row in info}
        if "profile" in column_names:
            return
        cur.execute(f"ALTER TABLE {table} RENAME TO {table}_old;")
        cur.execute(create_sql)
        existing_columns = [row[1] for row in info]
        copyable_columns = [col for col in copy_columns if col in existing_columns]
        if not copyable_columns:
            raise RuntimeError(
                f"Cannot migrate table {table}; no overlapping columns between legacy schema and destination."
            )
        cols = ", ".join(copyable_columns)
        cur.execute(
            f"""
            INSERT INTO {table} ({cols}, profile)
            SELECT {cols}, 'default' FROM {table}_old;
            """
        )
        cur.execute(f"DROP TABLE {table}_old;")

    def _ensure_columns(
        self,
        cur: sqlite3.Cursor,
        *,
        table: str,
        columns: dict[str, str],
    ) -> None:
        existing = {row[1] for row in cur.execute(f"PRAGMA table_info({table});").fetchall()}
        for name, definition in columns.items():
            if name not in existing:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition};")

    def reset_titles(self) -> None:
        """Clear titles table while cascading to dependents."""
        with self.cursor() as cur:
            cur.execute("DELETE FROM titles;")

    def bulk_insert_titles(self, records: Iterable[TitleRecord]) -> None:
        """Insert or replace title metadata."""
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
                (
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
                ),
            )

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
                INSERT INTO plots (tconst, status, source, content_hash, raw_path, clean_path, error, fetched_at)
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
        with self.cursor() as cur:
            row = cur.execute(
                """
                SELECT status, source, raw_path, clean_path, error, content_hash
                FROM plots
                WHERE tconst = ?
                """,
                (tconst,),
            ).fetchone()
        if not row:
            return None
        return PlotRecord(
            status=row[0],
            source=row[1],
            raw_path=row[2],
            clean_path=row[3],
            error=row[4],
            plot_hash=row[5],
        )

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
        active_profile = (profile or self.profile or "default").strip() or "default"
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

    def update_embedding_status(
        self,
        tconst: str,
        status: EmbeddingStatus,
        *,
        profile: str | None = None,
        session_id: str | None = None,
        model: str | None = None,
        prompt_hash: str | None = None,
        dim: int | None = None,
        input_tokens: int | None = None,
        vector_path: str | None = None,
        error: str | None = None,
    ) -> None:
        active_profile = (profile or self.profile or "default").strip() or "default"
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddings (
                    tconst,
                    profile,
                    status,
                    session_id,
                    model,
                    prompt_hash,
                    dim,
                    input_tokens,
                    vector_path,
                    error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tconst, profile) DO UPDATE SET
                    status=excluded.status,
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

    def iter_titles(self) -> Iterator[TitleRecord]:
        with self.cursor() as cur:
            for row in cur.execute(
                """
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
            ):
                yield _row_to_title(row)

    def ensure_analysis_record(
        self,
        record: TitleRecord,
        *,
        profile: str,
        plot_hash: str | None,
        path: Path,
        model: str,
    ) -> None:
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
                    attempts=COALESCE(analyses.attempts, 0)+1
                """,
                (
                    record.tconst,
                    profile,
                    "manual-import",
                    model,
                    plot_hash,
                    str(path),
                ),
            )
    def get_title(self, tconst: str) -> TitleRecord | None:
        with self.cursor() as cur:
            row = cur.execute(
                """
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
                WHERE tconst = ?
                """,
                (tconst,),
            ).fetchone()
        if not row:
            return None
        return _row_to_title(row)

    def search_titles(self, term: str, limit: int = 5) -> list[TitleRecord]:
        """Return titles whose names match the search term."""
        like_pattern = f"%{term.strip()}%"
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
                sort_rank ASC
            LIMIT ?
        """
        with self.cursor() as cur:
            rows = cur.execute(query, (like_pattern, term.strip(), limit)).fetchall()
        return [_row_to_title(row) for row in rows]

    def iter_analysis_tconsts(
        self,
        statuses: tuple[str, ...] | None = None,
        *,
        profile: str | None = None,
    ) -> Iterator[str]:
        """Yield tconst values from the analyses table filtered by status."""
        active_profile = (profile or self.profile or "default").strip() or "default"
        query = "SELECT tconst FROM analyses WHERE profile = ?"
        params: list[str] = [active_profile]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query = f"{query} AND status IN ({placeholders})"
            params.extend(statuses)
        with self.cursor() as cur:
            for row in cur.execute(query, params):
                yield row[0]

    def reset_analysis(self, tconst: str, *, profile: str | None = None) -> None:
        """Reset analysis status and metadata for a title."""
        active_profile = (profile or self.profile or "default").strip() or "default"
        with self.cursor() as cur:
            row = cur.execute(
                "SELECT 1 FROM analyses WHERE tconst = ? AND profile = ?", (tconst, active_profile)
            ).fetchone()
            plot_hash_row = cur.execute(
                "SELECT content_hash FROM plots WHERE tconst = ?", (tconst,)
            ).fetchone()
            plot_hash = plot_hash_row[0] if plot_hash_row else None
            if row:
                cur.execute(
                    """
                    UPDATE analyses
                    SET status = 'queued',
                        session_id = NULL,
                        model = NULL,
                        system_prompt_hash = NULL,
                        plot_hash = ?,
                        attempts = 0,
                        input_tokens = NULL,
                        input_cached_tokens = NULL,
                        output_tokens = NULL,
                        output_reasoning_tokens = NULL,
                        cost_estimate = NULL,
                        output_path = NULL,
                        error = NULL
                    WHERE tconst = ? AND profile = ?
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
                    ) VALUES (?, ?, 'queued', NULL, NULL, NULL, ?, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
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
        active_profile = (profile or self.profile or "default").strip() or "default"
        base_query = """
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
                COALESCE(a.status, ''),
                COALESCE(a.attempts, 0)
            FROM titles AS t
            JOIN plots AS p ON p.tconst = t.tconst AND p.status = 'ok'
            LEFT JOIN analyses AS a ON a.tconst = t.tconst AND a.profile = ?
        """
        params: list[object] = [active_profile]
        conditions: list[str] = []
        if not include_completed:
            conditions.append("COALESCE(a.status, '') != 'ok'")
        if conditions:
            base_query = f"{base_query} WHERE {' AND '.join(conditions)}"
        base_query = f"{base_query} ORDER BY t.sort_rank ASC"
        if limit is not None:
            base_query = f"{base_query} LIMIT ?"
            params.append(int(limit))
        with self.cursor() as cur:
            for row in cur.execute(base_query, params):
                title = TitleRecord(
                    tconst=row[0],
                    primary_title=row[1],
                    original_title=row[2],
                    title_type=row[3],
                    start_year=row[4],
                    end_year=row[5],
                    runtime_minutes=row[6],
                    genres=row[7],
                    num_votes=row[8],
                    average_rating=row[9],
                    sort_rank=row[10],
                )
                yield AnalysisCandidate(
                    title=title,
                    plot_source=row[11],
                    plot_path=Path(row[12]),
                    plot_hash=row[13],
                    analysis_status=row[14] or None,
                    attempts=row[15],
                )

    def iter_embedding_candidates(
        self,
        limit: int | None = None,
        *,
        include_completed: bool = False,
        profile: str | None = None,
    ) -> Iterator[EmbeddingCandidate]:
        active_profile = (profile or self.profile or "default").strip() or "default"
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
                a.status,
                COALESCE(e.status, '')
            FROM titles AS t
            JOIN analyses AS a ON a.tconst = t.tconst AND a.status = 'ok' AND a.profile = ?
            LEFT JOIN embeddings AS e ON e.tconst = t.tconst AND e.profile = ?
        """
        params: list[object] = [active_profile, active_profile]
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
                title = TitleRecord(
                    tconst=row[0],
                    primary_title=row[1],
                    original_title=row[2],
                    title_type=row[3],
                    start_year=row[4],
                    end_year=row[5],
                    runtime_minutes=row[6],
                    genres=row[7],
                    num_votes=row[8],
                    average_rating=row[9],
                    sort_rank=row[10],
                )
                output_path = Path(row[11])
                yield EmbeddingCandidate(
                    title=title,
                    analysis_path=output_path,
                    analysis_status=row[12] or None,
                    embedding_status=(row[13] or None),
                )

    def iter_sessions(self, limit: int = 10) -> Iterator[SessionRecord]:
        query = """
            SELECT
                id,
                component,
                started_at,
                completed_at,
                COALESCE(total_input_tokens, 0),
                COALESCE(total_cached_input_tokens, 0),
                COALESCE(total_output_tokens, 0),
                COALESCE(total_reasoning_tokens, 0),
                total_cost,
                notes
            FROM sessions
            ORDER BY started_at DESC
            LIMIT ?
        """
        with self.cursor() as cur:
            for row in cur.execute(query, (limit,)):
                yield SessionRecord(
                    id=row[0],
                    component=row[1],
                    started_at=row[2],
                    completed_at=row[3],
                    total_input_tokens=row[4],
                    total_cached_input_tokens=row[5],
                    total_output_tokens=row[6],
                    total_reasoning_tokens=row[7],
                    total_cost=row[8],
                    notes=row[9],
                )

    def session_totals(self) -> list[tuple[str, int, int, int, int, float]]:
        """Return aggregated token/cost totals per component."""
        query = """
            SELECT
                component,
                COALESCE(SUM(total_input_tokens), 0),
                COALESCE(SUM(total_cached_input_tokens), 0),
                COALESCE(SUM(total_output_tokens), 0),
                COALESCE(SUM(total_reasoning_tokens), 0),
                COALESCE(SUM(total_cost), 0.0)
            FROM sessions
            GROUP BY component
            ORDER BY component ASC
        """
        with self.cursor() as cur:
            return [tuple(row) for row in cur.execute(query)]

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
            LIMIT ?
        """
        with self.cursor() as cur:
            for row in cur.execute(query, (limit,)):
                yield _row_to_title(row)
ANALYSES_BASE_COLUMNS = [
    "tconst",
    "status",
    "session_id",
    "model",
    "system_prompt_hash",
    "plot_hash",
    "attempts",
    "input_tokens",
    "input_cached_tokens",
    "output_tokens",
    "output_reasoning_tokens",
    "cost_estimate",
    "output_path",
    "error",
    "updated_at",
]

ANALYSES_CREATE_SQL = """
CREATE TABLE analyses (
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
"""

ANALYSES_CREATE_SQL_IF_NOT_EXISTS = ANALYSES_CREATE_SQL.replace(
    "CREATE TABLE analyses", "CREATE TABLE IF NOT EXISTS analyses"
)

ANALYSES_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS trg_analyses_updated
AFTER UPDATE ON analyses
FOR EACH ROW
BEGIN
    UPDATE analyses
    SET updated_at = CURRENT_TIMESTAMP
    WHERE tconst = NEW.tconst AND profile = NEW.profile;
END;
"""

EMBEDDINGS_BASE_COLUMNS = [
    "tconst",
    "status",
    "session_id",
    "model",
    "prompt_hash",
    "dim",
    "input_tokens",
    "vector_path",
    "error",
    "updated_at",
]

EMBEDDINGS_CREATE_SQL = """
CREATE TABLE embeddings (
    tconst TEXT NOT NULL,
    profile TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL DEFAULT 'queued',
    session_id TEXT,
    model TEXT,
    prompt_hash TEXT,
    dim INTEGER,
    input_tokens INTEGER,
    vector_path TEXT,
    error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tconst, profile),
    FOREIGN KEY (tconst) REFERENCES titles (tconst) ON DELETE CASCADE
);
"""

EMBEDDINGS_CREATE_SQL_IF_NOT_EXISTS = EMBEDDINGS_CREATE_SQL.replace(
    "CREATE TABLE embeddings", "CREATE TABLE IF NOT EXISTS embeddings"
)

EMBEDDINGS_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS trg_embeddings_updated
AFTER UPDATE ON embeddings
FOR EACH ROW
BEGIN
    UPDATE embeddings
    SET updated_at = CURRENT_TIMESTAMP
    WHERE tconst = NEW.tconst AND profile = NEW.profile;
END;
"""

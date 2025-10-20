"""IMDb dataset ingestion utilities."""

from __future__ import annotations

import csv
import gzip
import hashlib
import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .manifest import Manifest, TitleRecord
from .paths import PATHS

LOGGER = logging.getLogger(__name__)

IMDB_BASE_URL = "https://datasets.imdbws.com"
DATASETS = {
    "title.basics": "title.basics.tsv.gz",
    "title.ratings": "title.ratings.tsv.gz",
}


@dataclass
class IMDbConfig:
    """Configuration for IMDb ingestion."""

    min_rating: float = 6.0
    min_votes: int = 10_000
    include_types: tuple[str, ...] = ("movie", "tvSeries", "tvMiniSeries")
    force_download: bool = False
    limit: int | None = None
    user_agent: str = "movie-pipeline/0.1 (+https://github.com/)"


def download_dataset(name: str, *, config: IMDbConfig) -> Path:
    """Download a compressed IMDb dataset if missing or force is set."""
    filename = DATASETS[name]
    dest = PATHS.imdb / filename
    if dest.exists() and not config.force_download:
        return dest

    url = f"{IMDB_BASE_URL}/{filename}"
    LOGGER.info("Downloading %s from %s", filename, url)
    request = urllib.request.Request(url, headers={"User-Agent": config.user_agent})
    start = time.time()
    with urllib.request.urlopen(request) as response, open(dest, "wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    LOGGER.info(
        "Finished downloading %s in %.2fs (%.2f MB)",
        filename,
        time.time() - start,
        dest.stat().st_size / 1_048_576,
    )
    return dest


def load_ratings(path: Path, *, config: IMDbConfig) -> dict[str, tuple[float, int]]:
    """Return ratings filtered by thresholds."""
    ratings: dict[str, tuple[float, int]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            tconst = row["tconst"]
            try:
                rating = float(row["averageRating"])
                votes = int(row["numVotes"])
            except (ValueError, KeyError):
                continue
            if votes >= config.min_votes and rating >= config.min_rating:
                ratings[tconst] = (rating, votes)
    LOGGER.info("Loaded %d ratings meeting thresholds", len(ratings))
    return ratings


def _parse_int(value: str) -> int | None:
    if not value or value == "\\N":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_str(value: str) -> str | None:
    if not value or value == "\\N":
        return None
    return value


def iter_filtered_titles(
    basics_path: Path,
    ratings: dict[str, tuple[float, int]],
    *,
    config: IMDbConfig,
) -> Iterator[TitleRecord]:
    """Yield TitleRecords matching filter criteria."""
    with gzip.open(basics_path, "rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            tconst = row["tconst"]
            if tconst not in ratings:
                continue
            title_type = row["titleType"]
            if title_type not in config.include_types:
                continue
            if row.get("isAdult") == "1":
                continue
            primary_title = row.get("primaryTitle") or ""
            original_title = _parse_str(row.get("originalTitle", ""))
            start_year = _parse_int(row.get("startYear", ""))
            end_year = _parse_int(row.get("endYear", ""))
            runtime_minutes = _parse_int(row.get("runtimeMinutes", ""))
            genres = _parse_str(row.get("genres", ""))
            average_rating, num_votes = ratings[tconst]
            yield TitleRecord(
                tconst=tconst,
                primary_title=primary_title,
                original_title=original_title,
                title_type=title_type,
                start_year=start_year,
                end_year=end_year,
                runtime_minutes=runtime_minutes,
                genres=genres,
                num_votes=num_votes,
                average_rating=average_rating,
                sort_rank=0,  # temporary, will be assigned later
            )


def ingest_imdb(manifest: Manifest, *, config: IMDbConfig) -> list[TitleRecord]:
    """Download IMDb datasets, filter titles, and write them to the manifest."""
    ratings_path = download_dataset("title.ratings", config=config)
    basics_path = download_dataset("title.basics", config=config)

    ratings = load_ratings(ratings_path, config=config)
    titles = list(iter_filtered_titles(basics_path, ratings, config=config))

    LOGGER.info("Collected %d candidate titles", len(titles))

    titles.sort(
        key=lambda r: (
            -r.average_rating,
            -r.num_votes,
            r.start_year if r.start_year is not None else 9999,
            r.tconst,
        )
    )
    if config.limit is not None:
        titles = titles[: config.limit]

    for idx, record in enumerate(titles):
        record.sort_rank = idx

    manifest.reset_titles()
    manifest.bulk_insert_titles(titles)

    def partial_checksum(path: Path, size: int = 1_000_000) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            digest.update(fh.read(size))
        return digest.hexdigest()

    metadata_path = PATHS.state / "imdb_snapshot.txt"
    checksum = hashlib.sha256(
        (partial_checksum(ratings_path) + partial_checksum(basics_path)).encode("utf-8")
    ).hexdigest()
    metadata_path.write_text(
        f"ratings_file={ratings_path.name}\n"
        f"basics_file={basics_path.name}\n"
        f"record_count={len(titles)}\n"
        f"min_rating={config.min_rating}\n"
        f"min_votes={config.min_votes}\n"
        f"include_types={','.join(config.include_types)}\n"
        f"limit={config.limit}\n"
        f"checksum_prefix={checksum}\n",
        encoding="utf-8",
    )

    LOGGER.info("Persisted %d titles into manifest", len(titles))
    return titles

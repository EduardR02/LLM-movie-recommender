"""Wikipedia plot acquisition utilities."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hashlib
import wikipediaapi

from .manifest import Manifest, TitleRecord
from .paths import PATHS

LOGGER = logging.getLogger(__name__)

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIPEDIA_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

PLOT_SECTION_NAMES = {
    "plot",
    "plot summary",
    "plot synopsis",
    "synopsis",
    "story",
    "summary",
    "premise",
}

EPISODE_SECTION_NAMES = {
    "episodes",
    "episode summary",
    "episode summaries",
    "episode list",
    "series overview",
    "season overview",
    "season summaries",
}

CHARACTER_SECTION_NAMES = {
    "characters",
    "main characters",
    "cast",
    "main cast",
    "principal cast",
}


@dataclass
class WikipediaConfig:
    """Configuration for Wikipedia fetching."""

    user_agent: str = "movie-pipeline/0.1 (+https://github.com/)"
    batch_size: int = 20
    max_requests_per_minute: int = 500
    throttle_seconds: float = 0.0
    language: str = "en"
    search_fallback: bool = False


def chunked(iterable: Iterable[str], size: int):
    chunk: list[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def throttle(rate_per_minute: int) -> float:
    if rate_per_minute <= 0:
        return 0.0
    return 60.0 / rate_per_minute


def call_wikidata(imdb_ids: list[str], *, config: WikipediaConfig) -> dict[str, str]:
    """Return mapping from IMDb IDs to English Wikipedia page titles."""
    throttle_delay = max(throttle(config.max_requests_per_minute), config.throttle_seconds)
    values = " ".join(f'"{mid}"' for mid in imdb_ids)
    query = f"""
    SELECT ?imdb ?title WHERE {{
      VALUES ?imdb {{ {values} }}
      ?item wdt:P345 ?imdb .
      ?sitelink schema:about ?item ;
                schema:isPartOf <https://en.wikipedia.org/> ;
                schema:name ?title .
    }}
    """
    params = urllib.parse.urlencode({"format": "json", "query": query})
    request = urllib.request.Request(
        f"{WIKIDATA_SPARQL_ENDPOINT}?{params}",
        headers={"User-Agent": config.user_agent},
    )
    if throttle_delay:
        time.sleep(throttle_delay)
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results = {}
    for binding in payload.get("results", {}).get("bindings", []):
        imdb_id = binding["imdb"]["value"]
        title = binding["title"]["value"]
        results[imdb_id] = title
    return results


def wiki_api_request(params: dict[str, str], *, config: WikipediaConfig) -> dict:
    encoded = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{WIKIPEDIA_API_ENDPOINT}?{encoded}",
        headers={"User-Agent": config.user_agent},
    )
    delay = max(throttle(config.max_requests_per_minute), config.throttle_seconds)
    if delay:
        time.sleep(delay)
    with urllib.request.urlopen(request) as response:
        data = response.read().decode("utf-8")
    return json.loads(data)


SECTION_PATTERN = re.compile(r"^==+\s*(.*?)\s*==+$", re.MULTILINE)


def search_wikipedia_titles(query: str, *, config: WikipediaConfig, limit: int = 5) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
    }
    payload = wiki_api_request(params, config=config)
    results = payload.get("query", {}).get("search", [])
    return [item.get("title", "") for item in results if item.get("title")]


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split plain-text article into (heading, content) sections."""
    if not text:
        return []
    matches = list(SECTION_PATTERN.finditer(text))
    sections: list[tuple[str | None, str]] = []
    if not matches:
        stripped = text.strip()
        if stripped:
            sections.append((None, stripped))
        return sections

    intro = text[: matches[0].start()].strip()
    if intro:
        sections.append((None, intro))

    for idx, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((heading, section_text))
    return sections


def select_relevant_sections(
    sections: list[tuple[str | None, str]],
    *,
    include_episodes: bool,
    include_characters: bool,
) -> list[tuple[str | None, str]]:
    """Pick sections that capture narrative essence."""
    if not sections:
        return []

    selected: list[tuple[str | None, str]] = []
    for heading, body in sections:
        if not body:
            continue
        if heading is None:
            # Always keep introduction.
            selected.append((heading, body))
            continue

        normalized = heading.lower()
        if normalized in PLOT_SECTION_NAMES:
            selected.append((heading, body))
        elif include_episodes and (
            normalized in EPISODE_SECTION_NAMES or normalized.startswith("season")
        ):
            selected.append((heading, body))
        elif include_characters and normalized in CHARACTER_SECTION_NAMES:
            selected.append((heading, body))

    if selected:
        return selected

    # Fallback: keep intro plus first couple of sections.
    intro = sections[0] if sections[0][0] is None else None
    remaining = sections[1:] if intro else sections
    fallback: list[tuple[str | None, str]] = []
    if intro:
        fallback.append(intro)
    fallback.extend(remaining[:2])
    return fallback


def format_sections(sections: list[tuple[str | None, str]]) -> str:
    """Render sections as structured text with headings."""
    parts: list[str] = []
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        if heading:
            parts.append(f"{heading.strip()}\n{body}")
        else:
            parts.append(body)
    text = "\n\n".join(parts)
    return clean_plot_text(text)


def clean_plot_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


MAIN_ARTICLE_PATTERN = re.compile(r"Main article[s]?:\s*(.+)")


def find_main_article_targets(text: str) -> list[str]:
    """Return titles referenced by 'Main article' templates."""
    targets: list[str] = []
    for match in MAIN_ARTICLE_PATTERN.finditer(text):
        remainder = match.group(1)
        first_line = remainder.splitlines()[0]
        candidates = re.split(r",|;| and ", first_line)
        for candidate in candidates:
            cleaned = candidate.strip(" .")
            if cleaned and cleaned not in targets:
                targets.append(cleaned)
    return targets


class PlotFetcher:
    """Fetch and persist plot summaries for IMDb titles."""

    def __init__(self, manifest: Manifest, config: WikipediaConfig | None = None) -> None:
        self.manifest = manifest
        self.config = config or WikipediaConfig()
        self._wiki = wikipediaapi.Wikipedia(
            user_agent=self.config.user_agent,
            language=self.config.language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

    def fetch_all(self, titles: Iterable[TitleRecord]) -> None:
        records = list(titles)
        if not records:
            LOGGER.warning("No titles supplied for plot fetching.")
            return

        records_by_id = {record.tconst: record for record in records}
        imdb_ids = [record.tconst for record in records]
        batches = chunked(imdb_ids, self.config.batch_size)
        for batch_num, batch in enumerate(batches, start=1):
            try:
                mapping = call_wikidata(batch, config=self.config)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed Wikidata lookup for batch %s: %s", batch_num, exc)
                for imdb_id in batch:
                    self.manifest.update_plot_status(
                        imdb_id, "error", error=f"wikidata_error:{exc}"
                    )
                continue

            for imdb_id in batch:
                title = mapping.get(imdb_id)
                if not title:
                    if self.config.search_fallback:
                        title = self._fallback_search(records_by_id[imdb_id])
                    if not title:
                        self.manifest.update_plot_status(
                            imdb_id, "missing", error="wikidata_missing"
                        )
                        continue
                try:
                    plot = self._fetch_plot(title, records_by_id[imdb_id])
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("Failed plot fetch for %s: %s", imdb_id, exc)
                    self.manifest.update_plot_status(
                        imdb_id, "error", source=title, error=f"wiki_fetch_error:{exc}"
                    )
                    continue
                if plot is None:
                    self.manifest.update_plot_status(
                        imdb_id, "missing", source=title, error="plot_section_missing"
                    )
                    continue
                content_hash = self._write_plot(imdb_id, plot)
                self.manifest.update_plot_status(
                    imdb_id,
                    "ok",
                    source=title,
                    content_hash=content_hash,
                    raw_path=str(self._raw_path(imdb_id)),
                    clean_path=str(self._clean_path(imdb_id)),
                )

    def _raw_path(self, tconst: str) -> Path:
        return PATHS.plots / f"{tconst}.raw.txt"

    def _clean_path(self, tconst: str) -> Path:
        return PATHS.plots / f"{tconst}.txt"

    def _write_plot(self, tconst: str, plot: str) -> str:
        raw_path = self._raw_path(tconst)
        clean_path = self._clean_path(tconst)
        raw_path.write_text(plot, encoding="utf-8")
        cleaned = clean_plot_text(plot)
        clean_path.write_text(cleaned, encoding="utf-8")
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        return digest

    def _fetch_plot(self, page_title: str, record: TitleRecord) -> str | None:
        cache_path = PATHS.cache / f"wiki_{slugify(page_title)}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = wiki_api_request(
                {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": "1",
                    "titles": page_title,
                    "redirects": "1",
                    "format": "json",
                },
                config=self.config,
            )
            cache_path.write_text(json.dumps(payload), encoding="utf-8")

        pages = payload.get("query", {}).get("pages", {})
        for _, page in pages.items():
            if "missing" in page:
                return None
            text = page.get("extract", "")
            if not text:
                continue
            is_series = record.title_type in {"tvSeries", "tvMiniSeries"}
            include_characters = is_series
            sections = split_sections(text)
            selected = select_relevant_sections(
                sections,
                include_episodes=is_series,
                include_characters=include_characters,
            )
            formatted = format_sections(selected) if selected else ""

            parts: list[str] = []
            if formatted:
                parts.append(formatted)

            # Some series defer details to dedicated episode pages; follow the main-article hint.
            if is_series and len("".join(parts)) < 500:
                for target in find_main_article_targets(text):
                    extra = self._fetch_episode_page(target)
                    if extra:
                        parts.append(extra)
                        break

            if not parts:
                parts.append(text.strip())

            combined = "\n\n".join(part for part in parts if part).strip()
            if combined:
                return combined
        return None

    def _fetch_episode_page(self, page_title: str) -> str | None:
        page = self._wiki.page(page_title)
        if not page.exists():
            return None
        sections = split_sections(page.text)
        selected = [
            (heading, body)
            for heading, body in sections
            if heading and heading.lower().startswith("season")
        ]
        if not selected:
            selected = select_relevant_sections(
                sections,
                include_episodes=True,
                include_characters=False,
            )
        formatted = format_sections(selected) if selected else ""
        return formatted or None

    def _fallback_search(self, record: TitleRecord) -> str | None:
        queries = [record.primary_title]
        if record.start_year:
            queries.append(f"{record.primary_title} ({record.start_year})")
        for query in queries:
            try:
                results = search_wikipedia_titles(query, config=self.config, limit=5)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Search fallback failed for %s: %s", query, exc)
                results = []
            for candidate in results:
                if candidate.lower().startswith(record.primary_title.lower()):
                    return candidate
            if results:
                return results[0]
        return None


def slugify(value: str) -> str:
    value = value.replace(" ", "_")
    value = re.sub(r"[^\w\-\.]", "", value)
    return value[:80]

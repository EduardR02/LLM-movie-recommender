"""Command-line interface for the movie pipeline."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Callable, Iterable

from .analysis import AnalysisConfig, AnalysisRunner
from .embeddings import (
    EmbeddingConfig,
    EmbeddingRunner,
    PlotEmbeddingRunner,
    OpenAIEmbeddingClient,
    QwenEmbeddingClient,
    SentenceTransformerEmbeddingClient,
    combine_embeddings,
    intersection_similar,
    load_embedding_vector,
    load_embeddings_matrix,
    resolve_embedding_instruction,
    is_qwen_embedding_model,
    top_k_similar,
)
from .imdb import IMDbConfig, ingest_imdb
from .logging_utils import configure_logging
from .manifest import Manifest, PlotRecord, TitleRecord
from .paths import PATHS, analyses_dir, embeddings_dir
from .plots import PlotFetcher, WikipediaConfig, slugify
from .explanations import (
    CandidateContext,
    SeedContext,
    SimilarityExplanationConfig,
    SimilarityExplainer,
)

LOGGER = logging.getLogger(__name__)

ACTIVE_PROFILE = "default"
EMBEDDING_INDEX_CHOICES = ("analysis", "plot")


def _resolve_embedding_dir(index: str, profile: str, *, embedding_set: str | None = None) -> Path:
    if index == "analysis":
        return embeddings_dir(profile, embedding_set)
    if index == "plot":
        PATHS.plot_embeddings.mkdir(parents=True, exist_ok=True)
        return PATHS.plot_embeddings
    raise KeyError(index)


def _format_title_label(record: TitleRecord) -> str:
    year = record.start_year or "????"
    try:
        rating = f"{record.average_rating:.1f}"
    except (TypeError, ValueError):
        rating = "n/a"
    return f"{record.primary_title} ({year}) — IMDb {rating}"


def _analysis_path(tconst: str) -> Path:
    return analyses_dir(ACTIVE_PROFILE) / f"{tconst}.txt"


def _read_plot_text(tconst: str, *, raw: bool = False) -> str | None:
    path = PATHS.plots / (f"{tconst}.raw.txt" if raw else f"{tconst}.txt")
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to read plot for %s: %s", tconst, exc)
        return None


def _read_analysis_text(tconst: str) -> str | None:
    path = _analysis_path(tconst)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to read analysis for %s: %s", tconst, exc)
        return None


def _resolve_title_identifier(
    manifest: Manifest,
    identifier: str,
    *,
    matches: int,
) -> TitleRecord:
    narrowed = identifier.strip()
    if not narrowed:
        raise SystemExit("Provide a non-empty title identifier.")
    if narrowed.lower().startswith("tt"):
        record = manifest.get_title(narrowed)
        if not record:
            raise SystemExit(f"No manifest record found for {narrowed}.")
        return record
    results = manifest.search_titles(narrowed, limit=matches)
    if not results:
        raise SystemExit(f"No titles matched '{narrowed}'.")
    record = results[0]
    year = record.start_year or "????"
    print(f"Resolved '{narrowed}' → {record.primary_title} ({year}) [{record.tconst}]")
    if len(results) > 1:
        suggestions = [
            f"{candidate.primary_title} ({candidate.start_year or '????'})"
            for candidate in results[1:]
        ]
        print("Other matches:", "; ".join(suggestions))
    return record


def _truncate_lines(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    clipped = lines[:max_lines]
    clipped.append("[...]")
    return "\n".join(clipped)


def _print_text_block(label: str, text: str | None, *, lines: int) -> None:
    print(f"\n-- {label}")
    if not text:
        print("  [Not available]")
        return
    snippet = _truncate_lines(text, lines)
    for line in snippet.splitlines():
        print(f"  {line}")


def _print_analysis_snippet(record: TitleRecord, *, lines: int) -> None:
    text = _read_analysis_text(record.tconst)
    header = _format_title_label(record)
    _print_text_block(header, text, lines=lines)


def _build_candidate_contexts(
    records: Iterable[tuple[TitleRecord, float]],
    *,
    limit: int,
) -> list[CandidateContext]:
    contexts: list[CandidateContext] = []
    for index, (record, score) in enumerate(records):
        if index >= limit:
            break
        analysis = _read_analysis_text(record.tconst)
        if not analysis:
            LOGGER.warning("Skipping explanation for %s; analysis missing.", record.tconst)
            continue
        contexts.append(
            CandidateContext(
                label=_format_title_label(record),
                score=score,
                summary=analysis,
            )
        )
    return contexts


def _select_explanation_records(
    records: list[tuple[TitleRecord, float]],
    *,
    top_k: int,
    indexes: list[int] | None,
) -> list[tuple[TitleRecord, float]]:
    if indexes:
        selected: list[tuple[TitleRecord, float]] = []
        for raw_index in indexes:
            if raw_index < 1 or raw_index > len(records):
                print(f"Ignoring --explain-index {raw_index}; out of range.")
                continue
            selected.append(records[raw_index - 1])
        return selected
    if top_k <= 0:
        return []
    return records[: min(top_k, len(records))]


def _generate_explanation(
    *,
    seed: SeedContext,
    records: list[tuple[TitleRecord, float]],
    top_k: int,
    indexes: list[int] | None = None,
) -> None:
    if not records:
        print("No recommendations to explain.")
        return
    selected_records = _select_explanation_records(records, top_k=top_k, indexes=indexes)
    if not selected_records:
        if indexes:
            print("No explanation candidates after applying --explain-index.")
        else:
            print("No explanation candidates; adjust --explain-top.")
        return
    contexts = _build_candidate_contexts(selected_records, limit=len(selected_records))
    if not contexts:
        print("No analyses available to explain these recommendations.")
        return
    try:
        explainer = SimilarityExplainer(config=SimilarityExplanationConfig())
    except Exception as exc:  # noqa: BLE001
        print(f"Explanations unavailable: {exc}")
        return
    try:
        explanation = explainer.explain(seed, contexts)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to generate explanation: {exc}")
        return
    print("\nWhy these recommendations:\n")
    print(explanation)


def _handle_compare(
    *,
    seed: TitleRecord | None,
    matches: list[tuple[TitleRecord, float]],
    indexes: list[int],
    lines: int,
) -> None:
    if not indexes:
        return
    selections: list[TitleRecord] = []
    for raw_index in indexes:
        if raw_index < 1 or raw_index > len(matches):
            print(f"Ignoring --compare index {raw_index}; out of range.")
            continue
        selections.append(matches[raw_index - 1][0])
    if not selections:
        return
    if seed:
        print("\nAnchor analysis:")
        _print_analysis_snippet(seed, lines=lines)
    print("\nSelected recommendation analyses:")
    for record in selections:
        _print_analysis_snippet(record, lines=lines)


def cmd_fetch_imdb(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    config = IMDbConfig(
        min_rating=args.min_rating,
        min_votes=args.min_votes,
        include_types=tuple(args.include_types),
        force_download=args.force_download,
        limit=args.limit,
        user_agent=args.user_agent,
    )
    titles = ingest_imdb(manifest, config=config)
    LOGGER.warning(
        "Ingested %d titles (rating >= %.1f, votes >= %d)",
        len(titles),
        config.min_rating,
        config.min_votes,
    )


def cmd_list_titles(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    for record in manifest.iter_titles():
        print(
            f"{record.sort_rank:05d} | {record.tconst} | "
            f"{record.primary_title} ({record.start_year or '????'}) | "
            f"{record.average_rating:.1f} ({record.num_votes} votes) | "
            f"{record.title_type}"
        )


def cmd_fetch_plots(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    fetcher = PlotFetcher(
        manifest,
        WikipediaConfig(
            user_agent=args.user_agent,
            batch_size=args.batch_size,
            max_requests_per_minute=args.max_requests,
            throttle_seconds=args.throttle,
            search_fallback=args.search_fallback,
        ),
    )
    titles = list(manifest.iter_titles())
    if args.limit is not None:
        titles = titles[: args.limit]
    LOGGER.info("Fetching plots for %d titles", len(titles))
    fetcher.fetch_all(titles)
    LOGGER.warning("Completed plot fetch for %d titles", len(titles))


def _collect_missing_plots(
    manifest: Manifest,
    *,
    limit: int | None = None,
) -> list[tuple[TitleRecord, PlotRecord | None]]:
    results: list[tuple[TitleRecord, PlotRecord | None]] = []
    for record in manifest.iter_titles():
        plot_info = manifest.get_plot_record(record.tconst)
        clean_path = PATHS.plots / f"{record.tconst}.txt"
        if plot_info is None or plot_info.status != "ok" or not clean_path.exists():
            results.append((record, plot_info))
            if limit is not None and len(results) >= limit:
                break
    return results


def cmd_verify_plots(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    pending_pairs = _collect_missing_plots(manifest, limit=args.limit)
    pending = [pair[0] for pair in pending_pairs]
    if not pending:
        print("All plots present and healthy.")
        return

    if args.refresh_cache:
        for record, info in pending_pairs:
            cache_keys: list[str] = []
            if info and info.source:
                cache_keys.append(info.source)
            cache_keys.append(record.primary_title)
            for key in cache_keys:
                slug = slugify(key)
                cache_path = PATHS.cache / f"wiki_{slug}.json"
                if cache_path.exists():
                    cache_path.unlink()

    fetcher = PlotFetcher(
        manifest,
        WikipediaConfig(
            user_agent=args.user_agent,
            batch_size=args.batch_size,
            max_requests_per_minute=args.max_requests,
            throttle_seconds=args.throttle,
            search_fallback=args.search_fallback,
        ),
    )

    LOGGER.info("Rebuilding plots for %d titles", len(pending))
    fetcher.fetch_all(pending)

    remaining_pairs = _collect_missing_plots(manifest)
    remaining_lookup = {record.tconst: info for record, info in remaining_pairs}
    still_missing: list[tuple[TitleRecord, PlotRecord | None]] = []
    for record, _ in pending_pairs:
        if record.tconst in remaining_lookup:
            still_missing.append((record, remaining_lookup.get(record.tconst)))

    if still_missing:
        print("Unable to rebuild the following plots:")
        for record, info in still_missing:
            year = record.start_year or "????"
            reason = (info.error if info and info.error else info.status if info else "missing")
            print(f"- {record.tconst} {record.primary_title} ({year}) [{reason}]")
    else:
        print("Successfully rebuilt plots for all requested titles.")


def cmd_run_analysis(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    default_prompt = PATHS.root / "prompts" / "system.txt"
    requested_prompt = Path(args.system_prompt)
    if args.profile != "default" and requested_prompt == default_prompt:
        candidate_names = [
            f"system_{args.profile}.txt",
            f"{args.profile}.txt",
        ]
        for name in candidate_names:
            profile_prompt = default_prompt.with_name(name)
            if profile_prompt.exists():
                LOGGER.info("Using profile-specific system prompt at %s", profile_prompt)
                requested_prompt = profile_prompt
                break
    config = AnalysisConfig(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        retry_limit=args.retry_limit,
        system_prompt_path=requested_prompt,
        max_concurrency=args.max_concurrency,
    )
    runner = AnalysisRunner(manifest, config=config)
    runner.run(limit=args.limit, force=args.force)


def cmd_compute_embeddings(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    provider = (args.provider or "openai").strip().lower()
    instruction = resolve_embedding_instruction(
        provider,
        args.model,
        args.instruction,
        for_query=False,
    )
    config = EmbeddingConfig(
        provider=provider,
        model=args.model,
        embedding_set=args.embedding_set,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        dimensions=args.dimensions if provider == "openai" else None,
        device=args.device,
        instruction=instruction,
    )
    runner = EmbeddingRunner(manifest, config=config)
    runner.run(limit=args.limit, force=args.force)


def cmd_compute_plot_embeddings(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    config = EmbeddingConfig(
        model=args.model,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        dimensions=args.dimensions,
    )
    runner = PlotEmbeddingRunner(manifest, config=config)
    runner.run(limit=args.limit, force=args.force)


def cmd_query_text(args: argparse.Namespace) -> None:
    try:
        directory = _resolve_embedding_dir(
            args.index,
            ACTIVE_PROFILE,
            embedding_set=args.embedding_set,
        )
    except KeyError:
        raise SystemExit(f"Unknown embedding index '{args.index}'.")

    ids, matrix, stored_dim = load_embeddings_matrix(directory)
    if matrix.size == 0:
        raise SystemExit("No embeddings available. Run compute-embeddings first.")

    target_dim = args.dimensions or stored_dim
    if target_dim != stored_dim:
        LOGGER.warning(
            "Stored %s embeddings use dimension %d but --dimensions=%d was requested.",
            args.index,
            stored_dim,
            target_dim,
        )

    provider = (args.provider or "openai").strip().lower()
    instruction = resolve_embedding_instruction(
        provider,
        args.model,
        args.instruction,
        for_query=True,
    )
    config = EmbeddingConfig(
        provider=provider,
        model=args.model,
        embedding_set=args.embedding_set,
        dimensions=target_dim if provider == "openai" else None,
        device=args.device,
        instruction=instruction,
        batch_size=1,
    )
    client = _build_embedding_client(config)
    embeddings, _ = client.embed([args.text])
    query_vector = embeddings[0]

    if len(query_vector) != stored_dim:
        raise SystemExit(
            f"Query embedding dimension ({len(query_vector)}) does not match stored vectors ({stored_dim}). "
            "Re-run compute-embeddings or adjust parameters."
        )

    results = top_k_similar(
        matrix,
        query_vector,
        ids,
        top_k=args.top_k,
        largest=not args.least_similar,
    )
    manifest = Manifest(profile=args.profile)
    collected: list[tuple[TitleRecord, float]] = []
    if args.least_similar and results:
        print("Least similar matches:")
    for imdb_id, score in results:
        title = manifest.get_title(imdb_id)
        if title:
            collected.append((title, score))
            label = _format_title_label(title)
        else:
            label = imdb_id
        print(f"{score:.4f} | {label}")

    if args.show_analysis and collected:
        print("\nAnalyses for recommended titles:")
        for record, _ in collected:
            _print_analysis_snippet(record, lines=args.analysis_lines)

    if getattr(args, "compare", None):
        _handle_compare(
            seed=None,
            matches=collected,
            indexes=args.compare,
            lines=args.analysis_lines,
        )

    if args.explain:
        seed = SeedContext(label=f"Free-text query: {args.text}", summary=args.text)
        _generate_explanation(
            seed=seed,
            records=collected,
            top_k=args.explain_top,
            indexes=args.explain_index or None,
        )


def cmd_query_title(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    identifiers = [identifier.strip() for identifier in args.identifiers if identifier.strip()]
    if not identifiers:
        raise SystemExit("Provide at least one title identifier.")

    try:
        directory = _resolve_embedding_dir(
            args.index,
            ACTIVE_PROFILE,
            embedding_set=args.embedding_set,
        )
    except KeyError:
        raise SystemExit(f"Unknown embedding index '{args.index}'.")

    ids, matrix, stored_dim = load_embeddings_matrix(directory)
    if matrix.size == 0:
        raise SystemExit("No embeddings available. Run compute-embeddings first.")

    seed_records: list[TitleRecord] = []
    vectors: list = []
    for identifier in identifiers:
        record = _resolve_title_identifier(
            manifest,
            identifier,
            matches=args.matches,
        )
        seed_records.append(record)
        try:
            vector, vector_dim = load_embedding_vector(record.tconst, vectors_dir=directory)
        except FileNotFoundError:
            raise SystemExit(
                f"Embedding not found for {record.tconst}. Run compute-embeddings first."
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(str(exc)) from exc
        if vector_dim != stored_dim:
            LOGGER.warning(
                "Vector dimension %d for %s differs from stored matrix dimension %d.",
                vector_dim,
                record.tconst,
                stored_dim,
            )
        vectors.append(vector)

    score_mode = args.score_mode
    weights = args.weights
    if score_mode == "intersection" and weights is not None:
        raise SystemExit("--weights can only be used with --score-mode centroid.")

    if score_mode == "centroid":
        if weights is not None:
            if len(weights) != len(seed_records):
                raise SystemExit("Number of --weights values must match the number of seeds.")
            if any(weight < 0 for weight in weights):
                raise SystemExit("Weights must be non-negative.")
            total = sum(weights)
            if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise SystemExit("Weights must sum to 1.0.")
        try:
            query_vector = combine_embeddings(vectors, weights=weights)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        results = top_k_similar(
            matrix,
            query_vector,
            ids,
            top_k=args.top_k,
            largest=not args.least_similar,
        )
    else:
        try:
            results = intersection_similar(
                matrix,
                vectors,
                ids,
                top_k=args.top_k,
                largest=not args.least_similar,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    collected: list[tuple[TitleRecord, float]] = []
    if args.least_similar and results:
        print("Least similar matches:")
    for imdb_id, score in results:
        title = manifest.get_title(imdb_id)
        if title:
            collected.append((title, score))
            label = _format_title_label(title)
        else:
            label = imdb_id
        print(f"{score:.4f} | {label}")

    if args.show_analysis:
        print("\nSeed analyses:")
        if seed_records:
            for record in seed_records:
                _print_analysis_snippet(record, lines=args.analysis_lines)
        else:
            print("  [Seed analysis unavailable]")
        if collected:
            print("\nAnalyses for recommended titles:")
            for record, _ in collected:
                _print_analysis_snippet(record, lines=args.analysis_lines)

    if getattr(args, "compare", None):
        anchor = seed_records[0] if len(seed_records) == 1 else None
        _handle_compare(
            seed=anchor,
            matches=collected,
            indexes=args.compare,
            lines=args.analysis_lines,
        )

    if args.explain:
        summaries: list[str] = []
        for record in seed_records:
            text = _read_analysis_text(record.tconst)
            if text:
                summaries.append(f"{_format_title_label(record)}\n{text}")
        if summaries:
            label_parts = [_format_title_label(record) for record in seed_records]
            seed = SeedContext(
                label=f"Combined seeds: {' + '.join(label_parts)}",
                summary="\n\n".join(summaries),
            )
            _generate_explanation(
                seed=seed,
                records=collected,
                top_k=args.explain_top,
                indexes=args.explain_index or None,
            )
        else:
            print("Seed analyses missing; cannot generate explanation.")


def cmd_show_title(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    identifier = args.identifier.strip()
    if not identifier:
        raise SystemExit("Provide a title name or IMDb ID.")
    if args.raw_plot and not args.plot:
        raise SystemExit("--raw-plot requires --plot.")

    record: TitleRecord | None = None
    if identifier.lower().startswith("tt"):
        record = manifest.get_title(identifier)
        if not record:
            raise SystemExit(f"No manifest record found for {identifier}.")
    else:
        matches = manifest.search_titles(identifier, limit=args.matches)
        if not matches:
            raise SystemExit(f"No titles matched '{identifier}'.")
        record = matches[0]
        year = record.start_year or "????"
        print(f"Resolved '{identifier}' → {record.primary_title} ({year}) [{record.tconst}]")
        if len(matches) > 1:
            suggestions = [
                f"{match.primary_title} ({match.start_year or '????'})"
                for match in matches[1:]
            ]
            print("Other matches:", "; ".join(suggestions))

    assert record is not None
    print(f"\nTitle: {_format_title_label(record)}")

    if not args.plot and not args.analysis:
        print("Use --plot and/or --analysis to display stored text.")
        return

    if args.plot:
        plot_text = _read_plot_text(record.tconst, raw=args.raw_plot)
        label = "Raw plot" if args.raw_plot else "Plot"
        _print_text_block(label, plot_text, lines=args.lines)

    if args.analysis:
        analysis_text = _read_analysis_text(record.tconst)
        _print_text_block("Analysis", analysis_text, lines=args.lines)


def _build_embedding_client(config: EmbeddingConfig):
    provider = (config.provider or "openai").strip().lower()
    if provider == "openai":
        return _build_openai_client(config)
    if provider in {"sentence-transformers", "huggingface", "hf"}:
        if is_qwen_embedding_model(config.model):
            return QwenEmbeddingClient(config)
        return SentenceTransformerEmbeddingClient(config)
    raise SystemExit(f"Unsupported embedding provider '{config.provider}'.")


def _build_openai_client(config: EmbeddingConfig) -> OpenAIEmbeddingClient:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY environment variable is required for queries.")
    return OpenAIEmbeddingClient(api_key, config)


def cmd_latest_titles(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    for record in manifest.iter_latest_titles(args.limit):
        year = record.start_year or "????"
        print(
            f"{record.tconst} | {record.primary_title} ({year}) | "
            f"{record.title_type} | rating {record.average_rating:.1f} | votes {record.num_votes}"
        )


def cmd_show_sessions(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    if args.aggregate:
        totals = manifest.session_totals()
        if not totals:
            print("No session data recorded yet.")
            return
        print("Aggregate usage by component:")
        for (
            component,
            prompt_tokens,
            cached_prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            cost,
        ) in totals:
            print(
                f"- {component}: prompt={prompt_tokens:,} (cached={cached_prompt_tokens:,}) "
                f"completion={completion_tokens:,} (reasoning={reasoning_tokens:,}) cost=${cost:,.2f}"
            )
        return

    sessions = list(manifest.iter_sessions(limit=args.limit))
    if not sessions:
        print("No sessions recorded yet.")
        return

    for session in sessions:
        completed = session.completed_at or "in-progress"
        notes = session.notes or ""
        print(
            f"{session.id} | {session.component} | started={session.started_at} "
            f"| completed={completed} | prompt={session.total_input_tokens:,} "
            f"(cached={session.total_cached_input_tokens:,}) "
            f"| completion={session.total_output_tokens:,} "
            f"(reasoning={session.total_reasoning_tokens:,}) "
            f"| cost=${session.total_cost:,.2f} "
            f"| notes={notes}"
        )


DEFAULT_CLEAR_STATUSES = ("ok", "error", "needs_retry")


def cmd_clear_analyses(args: argparse.Namespace) -> None:
    manifest = Manifest(profile=args.profile)
    if not args.all and not args.identifier:
        raise SystemExit("Provide --all or at least one --identifier to clear analyses.")

    statuses = tuple(args.status) if args.status else DEFAULT_CLEAR_STATUSES
    target_ids: set[str] = set()
    existing_with_status = set(
        manifest.iter_analysis_tconsts(statuses=statuses, profile=args.profile)
    )

    if args.all:
        target_ids.update(existing_with_status)

    identifiers = args.identifier or []
    for ident in identifiers:
        identifier = ident.strip()
        if not identifier:
            continue
        if identifier.lower().startswith("tt"):
            record = manifest.get_title(identifier)
            if record:
                target_ids.add(record.tconst)
                existing_with_status.add(record.tconst)
            else:
                LOGGER.warning("Skipping unknown IMDb ID %s", identifier)
        else:
            matches = manifest.search_titles(identifier, limit=1)
            if matches:
                record = matches[0]
                target_ids.add(record.tconst)
                existing_with_status.add(record.tconst)
                LOGGER.info(
                    "Resolved '%s' to %s (%s) [%s]",
                    identifier,
                    record.primary_title,
                    record.start_year or "????",
                    record.tconst,
                )
            else:
                LOGGER.warning("No title matched '%s'", identifier)

    if not target_ids:
        print("No analyses cleared.")
        return

    cleared = 0
    for tconst in sorted(target_ids):
        if args.all and statuses and existing_with_status and tconst not in existing_with_status:
            continue
        path = analyses_dir(args.profile) / f"{tconst}.txt"
        if path.exists():
            path.unlink()
        manifest.reset_analysis(tconst, profile=args.profile)
        cleared += 1

    print(f"Cleared analyses for {cleared} title(s).")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movie-pipeline",
        description="Utilities for generating rich movie embeddings.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (repeat for more detail).",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Profile name for prompt/embedding variants (default: default).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch-imdb", help="Download and filter IMDb datasets."
    )
    fetch_parser.add_argument("--min-rating", type=float, default=6.0)
    fetch_parser.add_argument("--min-votes", type=int, default=10_000)
    fetch_parser.add_argument(
        "--include-types",
        nargs="+",
        default=["movie", "tvSeries", "tvMiniSeries"],
        help="IMDb title types to include.",
    )
    fetch_parser.add_argument("--limit", type=int, default=None)
    fetch_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload source TSVs even if cached.",
    )
    fetch_parser.add_argument(
        "--user-agent",
        default="movie-pipeline/0.1 (+https://github.com/)",
        help="User agent used for IMDb downloads.",
    )
    fetch_parser.set_defaults(func=cmd_fetch_imdb)

    list_parser = subparsers.add_parser(
        "list-titles", help="Display titles stored in the manifest."
    )
    list_parser.set_defaults(func=cmd_list_titles)

    plots_parser = subparsers.add_parser(
        "fetch-plots", help="Fetch Wikipedia plot summaries for titles."
    )
    plots_parser.add_argument("--limit", type=int, default=None)
    plots_parser.add_argument("--batch-size", type=int, default=20)
    plots_parser.add_argument("--max-requests", type=int, default=500)
    plots_parser.add_argument("--throttle", type=float, default=0.0)
    plots_parser.add_argument(
        "--user-agent",
        default="movie-pipeline/0.1 (+https://github.com/)",
        help="User agent for Wikipedia/Wikidata requests.",
    )
    plots_parser.add_argument(
        "--search-fallback",
        action="store_true",
        help="Search Wikipedia directly when Wikidata has no sitelink.",
    )
    plots_parser.set_defaults(func=cmd_fetch_plots)

    verify_plots_parser = subparsers.add_parser(
        "verify-plots", help="Rebuild missing or failed Wikipedia plots."
    )
    verify_plots_parser.add_argument("--limit", type=int, default=None)
    verify_plots_parser.add_argument("--batch-size", type=int, default=20)
    verify_plots_parser.add_argument("--max-requests", type=int, default=500)
    verify_plots_parser.add_argument("--throttle", type=float, default=0.0)
    verify_plots_parser.add_argument(
        "--user-agent",
        default="movie-pipeline/0.1 (+https://github.com/)",
        help="User agent for Wikipedia/Wikidata requests.",
    )
    verify_plots_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Delete cached Wikipedia JSON before refetching.",
    )
    verify_plots_parser.add_argument(
        "--search-fallback",
        action="store_true",
        help="Search Wikipedia directly when Wikidata has no sitelink.",
    )
    verify_plots_parser.set_defaults(func=cmd_verify_plots)

    analysis_parser = subparsers.add_parser(
        "run-analysis", help="Generate Grok analyses for fetched plots."
    )
    analysis_parser.add_argument("--limit", type=int, default=None)
    analysis_parser.add_argument("--model", default="grok-4-fast-reasoning-latest")
    analysis_parser.add_argument("--max-output-tokens", type=int, default=16_000)
    analysis_parser.add_argument("--temperature", type=float, default=0.3)
    analysis_parser.add_argument("--top-p", type=float, default=None)
    analysis_parser.add_argument("--retry-limit", type=int, default=2)
    analysis_parser.add_argument(
        "--system-prompt",
        default=str((PATHS.root / "prompts" / "system.txt")),
        help="Path to the system prompt file.",
    )
    analysis_parser.add_argument("--max-concurrency", type=int, default=3)
    analysis_parser.add_argument("--dry-run", action="store_true")
    analysis_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate analyses even if they already exist.",
    )
    analysis_parser.set_defaults(func=cmd_run_analysis)

    embed_parser = subparsers.add_parser(
        "compute-embeddings", help="Generate embeddings for Grok analyses."
    )
    embed_parser.add_argument("--limit", type=int, default=None)
    embed_parser.add_argument(
        "--provider",
        choices=("openai", "sentence-transformers"),
        default="openai",
        help="Embedding provider: 'openai' or local 'sentence-transformers'.",
    )
    embed_parser.add_argument("--model", default="text-embedding-3-large")
    embed_parser.add_argument(
        "--embedding-set",
        default="default",
        help="Name for embedding variant directory (default: default).",
    )
    embed_parser.add_argument("--batch-size", type=int, default=32)
    embed_parser.add_argument("--dry-run", action="store_true")
    embed_parser.add_argument(
        "--dimensions",
        type=int,
        default=1024,
        help="Optional embedding dimensionality (e.g., 1024 or 3072).",
    )
    embed_parser.add_argument(
        "--device",
        default=None,
        help="Device identifier for local models (e.g., cuda, cuda:0, cpu).",
    )
    embed_parser.add_argument(
        "--instruction",
        default=None,
        help=(
            "Optional instruction prefix or template for local embeddings. "
            "Use '{text}' placeholder to control formatting."
        ),
    )
    embed_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even if they already exist.",
    )
    embed_parser.set_defaults(func=cmd_compute_embeddings)

    plot_embed_parser = subparsers.add_parser(
        "compute-plot-embeddings",
        help="Generate embeddings directly from cleaned plot texts.",
    )
    plot_embed_parser.add_argument("--limit", type=int, default=None)
    plot_embed_parser.add_argument("--model", default="text-embedding-3-large")
    plot_embed_parser.add_argument("--batch-size", type=int, default=32)
    plot_embed_parser.add_argument("--dry-run", action="store_true")
    plot_embed_parser.add_argument(
        "--dimensions",
        type=int,
        default=1024,
        help="Optional embedding dimensionality (e.g., 1024 or 3072).",
    )
    plot_embed_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute plot embeddings even if they already exist.",
    )
    plot_embed_parser.set_defaults(func=cmd_compute_plot_embeddings)

    query_text_parser = subparsers.add_parser(
        "query-text", help="Find similar titles for a free-text query."
    )
    query_text_parser.add_argument("text")
    query_text_parser.add_argument(
        "--index",
        default="analysis",
        choices=EMBEDDING_INDEX_CHOICES,
        help="Embedding index to search (analysis or plot).",
    )
    query_text_parser.add_argument(
        "--embedding-set",
        default="default",
        help="Embedding variant directory to search (default: default).",
    )
    query_text_parser.add_argument(
        "--provider",
        choices=("openai", "sentence-transformers"),
        default="openai",
        help="Provider to use for embedding the query text.",
    )
    query_text_parser.add_argument("--model", default="text-embedding-3-large")
    query_text_parser.add_argument("--top-k", type=int, default=10)
    query_text_parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Override embedding dimensionality (defaults to stored vector size).",
    )
    query_text_parser.add_argument(
        "--device",
        default=None,
        help="Device override for local query embeddings (e.g., cuda).",
    )
    query_text_parser.add_argument(
        "--instruction",
        default=None,
        help="Instruction prefix/template when using local embeddings.",
    )
    query_text_parser.add_argument(
        "--show-analysis",
        action="store_true",
        help="Display Grok analysis snippets for recommended titles.",
    )
    query_text_parser.add_argument(
        "--analysis-lines",
        type=int,
        default=20,
        help="Number of lines to show from each analysis when --show-analysis is used.",
    )
    query_text_parser.add_argument(
        "--compare",
        type=int,
        action="append",
        default=[],
        help="Show analysis snippets for specific recommendation numbers (1-based).",
    )
    query_text_parser.add_argument(
        "--explain",
        action="store_true",
        help="Ask Grok to explain why the top matches relate to the query.",
    )
    query_text_parser.add_argument(
        "--explain-top",
        type=int,
        default=3,
        help="Number of top matches to include in the explanation request.",
    )
    query_text_parser.add_argument(
        "--explain-index",
        type=int,
        action="append",
        default=None,
        help="Only include these recommendation numbers (1-based) in the explanation.",
    )
    query_text_parser.add_argument(
        "--least-similar",
        action="store_true",
        help="Return the least similar matches instead of the closest ones.",
    )
    query_text_parser.set_defaults(func=cmd_query_text)

    query_title_parser = subparsers.add_parser(
        "query-title", help="Find similar titles to one or more seed titles."
    )
    query_title_parser.add_argument(
        "identifiers",
        nargs="+",
        help="IMDb IDs (tt...) or title text. Provide multiple values to blend them.",
    )
    query_title_parser.add_argument("--top-k", type=int, default=10)
    query_title_parser.add_argument(
        "--matches",
        type=int,
        default=5,
        help="Number of title matches to show when resolving by name.",
    )
    query_title_parser.add_argument(
        "--index",
        default="analysis",
        choices=EMBEDDING_INDEX_CHOICES,
        help="Embedding index to search (analysis or plot).",
    )
    query_title_parser.add_argument(
        "--embedding-set",
        default="default",
        help="Embedding variant directory to use (default: default).",
    )
    query_title_parser.add_argument(
        "--show-analysis",
        action="store_true",
        help="Display Grok analysis snippets for the seed and recommended titles.",
    )
    query_title_parser.add_argument(
        "--analysis-lines",
        type=int,
        default=20,
        help="Number of lines to show from each analysis when --show-analysis is used.",
    )
    query_title_parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Weights for each seed (must sum to 1.0).",
    )
    query_title_parser.add_argument(
        "--score-mode",
        choices=("centroid", "intersection"),
        default="centroid",
        help="Blend seeds via centroid (default) or require overlap with intersection scoring.",
    )
    query_title_parser.add_argument(
        "--compare",
        type=int,
        action="append",
        default=[],
        help="Compare the seed analysis with specific recommendation numbers (1-based).",
    )
    query_title_parser.add_argument(
        "--explain",
        action="store_true",
        help="Ask Grok to explain why the top matches relate to the seed title.",
    )
    query_title_parser.add_argument(
        "--explain-top",
        type=int,
        default=3,
        help="Number of top matches to include in the explanation request.",
    )
    query_title_parser.add_argument(
        "--explain-index",
        type=int,
        action="append",
        default=None,
        help="Only include these recommendation numbers (1-based) in the explanation.",
    )
    query_title_parser.add_argument(
        "--least-similar",
        action="store_true",
        help="Return the least similar matches instead of the closest ones.",
    )
    query_title_parser.set_defaults(func=cmd_query_title)

    show_title_parser = subparsers.add_parser(
        "show-title", help="Display stored plot and/or analysis text for a title."
    )
    show_title_parser.add_argument("identifier", help="IMDb ID (tt...) or title text.")
    show_title_parser.add_argument(
        "--matches",
        type=int,
        default=5,
        help="Number of title matches to show when resolving by name.",
    )
    show_title_parser.add_argument(
        "--plot",
        action="store_true",
        help="Print the stored plot text (cleaned).",
    )
    show_title_parser.add_argument(
        "--raw-plot",
        action="store_true",
        help="Use the raw plot text instead of the cleaned version (requires --plot).",
    )
    show_title_parser.add_argument(
        "--analysis",
        action="store_true",
        help="Print the stored Grok analysis text.",
    )
    show_title_parser.add_argument(
        "--lines",
        type=int,
        default=0,
        help="Limit the number of lines printed (0 shows the full text).",
    )
    show_title_parser.set_defaults(func=cmd_show_title)

    latest_parser = subparsers.add_parser(
        "latest-titles", help="List newest titles in the manifest."
    )
    latest_parser.add_argument("--limit", type=int, default=10)
    latest_parser.set_defaults(func=cmd_latest_titles)

    sessions_parser = subparsers.add_parser(
        "sessions", help="Show Grok/OpenAI usage sessions."
    )
    sessions_parser.add_argument("--limit", type=int, default=10)
    sessions_parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Display aggregated totals per component.",
    )
    sessions_parser.set_defaults(func=cmd_show_sessions)

    clear_parser = subparsers.add_parser(
        "clear-analyses", help="Reset Grok analyses and delete outputs."
    )
    clear_parser.add_argument(
        "--identifier",
        action="append",
        help="IMDb ID (tt...) or title text to clear. Repeat for multiple titles.",
    )
    clear_parser.add_argument(
        "--all",
        action="store_true",
        help="Clear analyses for all titles in the manifest.",
    )
    clear_parser.add_argument(
        "--status",
        action="append",
        choices=["queued", "running", "ok", "error", "needs_retry"],
        help="Only clear analyses currently in these statuses (default: ok, error, needs_retry).",
    )
    clear_parser.set_defaults(func=cmd_clear_analyses)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    profile = (getattr(args, "profile", "default") or "default").strip() or "default"
    args.profile = profile
    global ACTIVE_PROFILE
    ACTIVE_PROFILE = profile
    configure_logging(args.verbose)
    func: Callable[[argparse.Namespace], None] = getattr(args, "func")
    func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

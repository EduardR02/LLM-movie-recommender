"""LLM analysis pipeline supporting both Grok (xAI) and DeepSeek."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from openai import AsyncOpenAI
from xai_sdk import AsyncClient
from xai_sdk.chat import system as grok_system
from xai_sdk.chat import user as grok_user

try:  # gRPC only available when running against Grok; fallback gracefully otherwise.
    import grpc
except Exception:  # noqa: BLE001
    grpc = None

from .manifest import AnalysisCandidate, Manifest, TitleRecord
from .paths import PATHS, analyses_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    model: str = "grok-4-fast-reasoning-latest"
    max_output_tokens: int = 16_000
    temperature: float = 0.8
    top_p: float | None = None
    retry_limit: int = 2
    system_prompt_path: Path = PATHS.root / "prompts" / "system.txt"
    timeout_seconds: int = 600
    max_concurrency: int = 3
    input_cost_per_mtoken: float = 0.20
    output_cost_per_mtoken: float = 0.50
    cached_input_cost_per_mtoken: float | None = None  # If None, use input_cost


@dataclass
class DeepSeekConfig:
    model: str = "deepseek-reasoner"
    max_output_tokens: int = 16_000
    temperature: float = 1.2
    top_p: float | None = None
    retry_limit: int = 2
    system_prompt_path: Path = PATHS.root / "prompts" / "system.txt"
    timeout_seconds: int = 600
    max_concurrency: int = 3
    input_cost_per_mtoken: float = 0.28
    cached_input_cost_per_mtoken: float = 0.028
    output_cost_per_mtoken: float = 0.42


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    prompt_cached_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    prompt_cached_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, result: LLMResult) -> None:
        self.prompt_tokens += result.prompt_tokens
        self.prompt_cached_tokens += result.prompt_cached_tokens
        self.completion_tokens += result.completion_tokens
        self.total_tokens += result.total_tokens
        self.reasoning_tokens += result.reasoning_tokens


def _is_rate_limited_error(exc: Exception) -> bool:
    """Return True if the exception represents a rate limit (Grok or DeepSeek)."""
    # Check for Grok gRPC rate limits
    if grpc is not None:
        status_enum = getattr(grpc, "StatusCode", None)
        aio_module = getattr(grpc, "aio", None)
        rpc_error_cls = getattr(aio_module, "AioRpcError", None) if aio_module else None
        if status_enum and rpc_error_cls and isinstance(exc, rpc_error_cls):
            try:
                return exc.code() == status_enum.RESOURCE_EXHAUSTED
            except Exception:  # noqa: BLE001
                pass

    # Check for OpenAI/DeepSeek rate limit errors
    exc_type_name = type(exc).__name__
    if "RateLimitError" in exc_type_name:
        return True

    # Check message for common rate limit indicators
    message = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in message
        or "Too many requests" in message
        or "rate_limit" in message.lower()
        or "429" in message
    )


class AnalysisRunner:
    """Generate rich analyses for plots using LLM providers (Grok or DeepSeek)."""

    def __init__(
        self,
        manifest: Manifest,
        config: AnalysisConfig | DeepSeekConfig | None = None,
        provider: str = "grok",
    ) -> None:
        self.manifest = manifest
        self.provider = provider.lower()

        # Set default config based on provider
        if config is None:
            config = DeepSeekConfig() if self.provider == "deepseek" else AnalysisConfig()
        self.config = config

        profile = getattr(self.manifest, "profile", None)
        if profile is None or not str(profile).strip():
            raise ValueError("Manifest.profile must be set before running analyses.")
        self.profile = str(profile).strip()
        self.analysis_dir = analyses_dir(self.profile)

        # Set up API key based on provider
        if self.provider == "grok":
            self.api_key = os.environ.get("XAI_API_KEY")
            if not self.api_key:
                raise RuntimeError("XAI_API_KEY environment variable is required for Grok analysis.")
        elif self.provider == "deepseek":
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise RuntimeError("DEEPSEEK_API_KEY environment variable is required for DeepSeek analysis.")
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Must be 'grok' or 'deepseek'.")

    def run(self, *, limit: int | None = None, force: bool = False) -> None:
        candidates = list(
            self.manifest.iter_analysis_candidates(
                limit,
                include_completed=force,
                profile=self.profile,
            )
        )
        if not candidates:
            LOGGER.warning("No analysis candidates found.")
            return

        if force:
            LOGGER.info(
                "Force flag enabled; preparing %d analyses for regeneration.",
                len(candidates),
            )
            for candidate in candidates:
                self._prepare_for_regeneration(candidate)

        system_prompt = self._load_system_prompt(self.config.system_prompt_path)
        system_prompt_hash = sha256(system_prompt.encode("utf-8")).hexdigest()

        session_id = f"{self.provider}-{uuid.uuid4().hex[:10]}"
        component_name = f"{self.provider}_analysis[{self.profile}]"
        self.manifest.register_session(session_id, component=component_name)

        LOGGER.info(
            "Starting %s analysis for %d titles (session=%s)",
            self.provider.capitalize(),
            len(candidates),
            session_id,
        )
        totals = UsageTotals()
        try:
            totals = self._run_event_loop(
                self._run_async(
                    candidates=candidates,
                    system_prompt=system_prompt,
                    system_prompt_hash=system_prompt_hash,
                    session_id=session_id,
                )
            )
        finally:
            notes = f"profile={self.profile} titles={len(candidates)}"
            total_cost = self._estimate_cost(
                totals.prompt_tokens, totals.completion_tokens, totals.prompt_cached_tokens
            )
            self.manifest.finalize_session(
                session_id,
                total_input_tokens=totals.prompt_tokens,
                total_cached_input_tokens=totals.prompt_cached_tokens,
                total_output_tokens=totals.completion_tokens,
                total_reasoning_tokens=totals.reasoning_tokens,
                total_cost=total_cost,
                notes=notes,
            )
            LOGGER.info(
                "%s analysis session complete: prompt=%d (cached=%d) completion=%d (reasoning=%d)",
                self.provider.capitalize(),
                totals.prompt_tokens,
                totals.prompt_cached_tokens,
                totals.completion_tokens,
                totals.reasoning_tokens,
            )

    def _run_event_loop(self, coro: asyncio.Future) -> UsageTotals:
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:
            if "asyncio.run() cannot be called" not in str(exc):
                raise
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)

    async def _run_async(
        self,
        *,
        candidates: Iterable[AnalysisCandidate],
        system_prompt: str,
        system_prompt_hash: str,
        session_id: str,
    ) -> UsageTotals:
        # Create the appropriate client based on provider
        if self.provider == "grok":
            client = AsyncClient(api_key=self.api_key, timeout=self.config.timeout_seconds)
        elif self.provider == "deepseek":
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                timeout=self.config.timeout_seconds,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        totals = UsageTotals()
        totals_lock = asyncio.Lock()

        async def process(candidate: AnalysisCandidate) -> None:
            async with semaphore:
                await self._process_candidate(
                    client=client,
                    candidate=candidate,
                    system_prompt=system_prompt,
                    system_prompt_hash=system_prompt_hash,
                    session_id=session_id,
                    totals=totals,
                    totals_lock=totals_lock,
                )

        await asyncio.gather(*(process(candidate) for candidate in candidates))
        return totals

    async def _process_candidate(
        self,
        *,
        client: AsyncClient,
        candidate: AnalysisCandidate,
        system_prompt: str,
        system_prompt_hash: str,
        session_id: str,
        totals: UsageTotals,
        totals_lock: asyncio.Lock,
    ) -> None:
        title = candidate.title
        tconst = title.tconst
        plot_path = candidate.plot_path
        if not plot_path.exists():
            LOGGER.error("Plot file missing for %s (%s)", title.primary_title, plot_path)
            self.manifest.update_analysis_status(
                tconst,
                "error",
                profile=self.profile,
                session_id=session_id,
                model=self.config.model,
                plot_hash=candidate.plot_hash,
                error="plot_missing",
            )
            return

        try:
            plot_text = plot_path.read_text(encoding="utf-8").strip()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to read plot for %s: %s", title.primary_title, exc)
            self.manifest.update_analysis_status(
                tconst,
                "error",
                profile=self.profile,
                session_id=session_id,
                model=self.config.model,
                plot_hash=candidate.plot_hash,
                error="plot_read_error",
            )
            return

        if not plot_text:
            LOGGER.warning("Empty plot text for %s; skipping Grok analysis", title.primary_title)
            self.manifest.update_analysis_status(
                tconst,
                "error",
                profile=self.profile,
                session_id=session_id,
                model=self.config.model,
                plot_hash=candidate.plot_hash,
                error="plot_empty",
            )
            return

        max_attempts = self.config.retry_limit + 1
        for attempt_index in range(max_attempts):
            current_attempt = candidate.attempts + attempt_index + 1
            self.manifest.update_analysis_status(
                tconst,
                "running",
                profile=self.profile,
                session_id=session_id,
                model=self.config.model,
                system_prompt_hash=system_prompt_hash,
                plot_hash=candidate.plot_hash,
                attempts=current_attempt,
                error=None,
            )
            try:
                result = await self._invoke_model(
                    client=client,
                    system_prompt=system_prompt,
                    formatted_input=self._format_input(candidate.title, plot_text),
                )
            except Exception as exc:  # noqa: BLE001
                rate_limited = _is_rate_limited_error(exc)
                LOGGER.error(
                    "%s request failed for %s (attempt %d/%d): %s",
                    self.provider.capitalize(),
                    title.primary_title,
                    attempt_index + 1,
                    max_attempts,
                    exc,
                )
                status = "needs_retry" if attempt_index + 1 < max_attempts else "error"
                self.manifest.update_analysis_status(
                    tconst,
                    status,
                    profile=self.profile,
                    session_id=session_id,
                    model=self.config.model,
                    system_prompt_hash=system_prompt_hash,
                    plot_hash=candidate.plot_hash,
                    attempts=current_attempt,
                    error=str(exc),
                )
                if attempt_index + 1 >= max_attempts:
                    return
                delay = 10 if rate_limited else min(2 ** attempt_index, 10)
                if rate_limited:
                    LOGGER.warning(
                        "Rate limit hit; sleeping %.1f seconds before retrying %s",
                        delay,
                        title.primary_title,
                    )
                await asyncio.sleep(delay)
                continue

            output_path = self._write_analysis(tconst, result.text)
            self.manifest.update_analysis_status(
                tconst,
                "ok",
                profile=self.profile,
                session_id=session_id,
                model=self.config.model,
                system_prompt_hash=system_prompt_hash,
                plot_hash=candidate.plot_hash,
                attempts=current_attempt,
                input_tokens=result.prompt_tokens,
                input_cached_tokens=result.prompt_cached_tokens,
                output_tokens=result.completion_tokens,
                output_reasoning_tokens=result.reasoning_tokens,
                cost_estimate=self._estimate_cost(
                    result.prompt_tokens, result.completion_tokens, result.prompt_cached_tokens
                ),
                output_path=str(output_path),
                error=None,
            )
            async with totals_lock:
                totals.add(result)
            LOGGER.info("Analysis complete for %s", title.primary_title)
            return

    async def _invoke_model(
        self,
        *,
        client: AsyncClient | AsyncOpenAI,
        system_prompt: str,
        formatted_input: str,
    ) -> LLMResult:
        if self.provider == "grok":
            return await self._invoke_grok(client, system_prompt, formatted_input)
        elif self.provider == "deepseek":
            return await self._invoke_deepseek(client, system_prompt, formatted_input)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _invoke_grok(
        self,
        client: AsyncClient,
        system_prompt: str,
        formatted_input: str,
    ) -> LLMResult:
        create_kwargs: dict[str, object] = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.top_p is not None:
            create_kwargs["top_p"] = self.config.top_p

        messages = [grok_system(system_prompt)] if system_prompt else []
        chat = client.chat.create(messages=messages, **create_kwargs)
        chat.append(grok_user(formatted_input))

        response = await chat.sample()
        usage = getattr(response, "usage", None)

        def _get_attr(obj: object, attr: str) -> object:
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(attr)
            return getattr(obj, attr, None)

        def _maybe_int(value: object) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(float(value))
                except ValueError:
                    return None
            return None

        prompt_tokens = _maybe_int(_get_attr(usage, "prompt_tokens")) or 0
        completion_tokens = _maybe_int(_get_attr(usage, "completion_tokens")) or 0
        total_tokens = _maybe_int(_get_attr(usage, "total_tokens"))
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        prompt_details = _get_attr(usage, "prompt_tokens_details")
        prompt_cached_tokens = _maybe_int(_get_attr(prompt_details, "cached_tokens"))
        if prompt_cached_tokens is None:
            prompt_cached_tokens = 0

        completion_details = _get_attr(usage, "completion_tokens_details")
        reasoning_tokens = _maybe_int(_get_attr(completion_details, "reasoning_tokens"))
        if reasoning_tokens is None:
            reasoning_tokens = _maybe_int(_get_attr(usage, "reasoning_tokens")) or 0

        content = getattr(response, "content", "")
        if not isinstance(content, str):
            raise RuntimeError("Unexpected Grok response payload.")

        return LLMResult(
            text=content.strip(),
            prompt_tokens=prompt_tokens,
            prompt_cached_tokens=prompt_cached_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    async def _invoke_deepseek(
        self,
        client: AsyncOpenAI,
        system_prompt: str,
        formatted_input: str,
    ) -> LLMResult:
        """Invoke DeepSeek API using OpenAI SDK."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": formatted_input})

        create_kwargs: dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.top_p is not None:
            create_kwargs["top_p"] = self.config.top_p

        response = await client.chat.completions.create(**create_kwargs)

        # Extract usage information
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)

        # DeepSeek supports prompt caching
        prompt_cached_tokens = 0
        if usage and hasattr(usage, "prompt_cache_hit_tokens"):
            prompt_cached_tokens = usage.prompt_cache_hit_tokens or 0

        # Extract reasoning tokens (DeepSeek reasoner includes this)
        reasoning_tokens = 0
        if usage and hasattr(usage, "completion_tokens_details"):
            details = usage.completion_tokens_details
            if details and hasattr(details, "reasoning_tokens"):
                reasoning_tokens = details.reasoning_tokens or 0

        # Extract the content - for DeepSeek reasoner, we only want the final content,
        # not the reasoning trace
        choice = response.choices[0]
        content = choice.message.content or ""

        return LLMResult(
            text=content.strip(),
            prompt_tokens=prompt_tokens,
            prompt_cached_tokens=prompt_cached_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def _format_input(self, title: TitleRecord, plot_text: str) -> str:
        """Compose the user message with title metadata and plot."""
        year = title.start_year or "????"
        header = f"Title: {title.primary_title} ({year})"
        return f"{header}\n\nPlot:\n{plot_text}"

    def _write_analysis(self, tconst: str, text: str) -> Path:
        path = self.analysis_dir / f"{tconst}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def _prepare_for_regeneration(self, candidate: AnalysisCandidate) -> None:
        """Delete existing analysis artifacts and reset manifest status."""
        tconst = candidate.title.tconst
        existing_path = self.analysis_dir / f"{tconst}.txt"
        if existing_path.exists():
            existing_path.unlink()
        self.manifest.reset_analysis(tconst, profile=self.profile)

    def _estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> float:
        """Estimate cost accounting for cached tokens if applicable."""
        # Calculate non-cached prompt tokens
        uncached_prompt_tokens = max(0, prompt_tokens - cached_tokens)

        # Cost for uncached prompt tokens
        prompt_cost = (uncached_prompt_tokens / 1_000_000) * self.config.input_cost_per_mtoken

        # Cost for cached tokens (if provider supports it)
        cached_cost = 0.0
        if cached_tokens > 0 and hasattr(self.config, "cached_input_cost_per_mtoken"):
            cached_rate = self.config.cached_input_cost_per_mtoken
            if cached_rate is not None:
                cached_cost = (cached_tokens / 1_000_000) * cached_rate

        # Cost for completion tokens
        completion_cost = (completion_tokens / 1_000_000) * self.config.output_cost_per_mtoken

        return round(prompt_cost + cached_cost + completion_cost, 6)

    @staticmethod
    def _load_system_prompt(path: Path) -> str:
        if not path.exists():
            LOGGER.warning("System prompt file missing at %s; using empty prompt.", path)
            return ""
        return path.read_text(encoding="utf-8")

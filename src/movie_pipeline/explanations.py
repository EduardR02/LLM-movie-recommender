"""Explain embedding-based recommendations using Grok analyses."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from xai_sdk import AsyncClient
from xai_sdk.chat import system as grok_system
from xai_sdk.chat import user as grok_user


@dataclass
class SimilarityExplanationConfig:
    """Configuration for similarity explanations."""

    model: str = "grok-4-fast-reasoning-latest"
    max_output_tokens: int = 2048
    temperature: float = 0.3
    top_p: float | None = None
    timeout_seconds: int = 120
    max_seed_chars: int = 4000
    max_candidate_chars: int = 3000


@dataclass
class SeedContext:
    label: str
    summary: str


@dataclass
class CandidateContext:
    label: str
    score: float
    summary: str


class SimilarityExplainer:
    """Generate natural-language rationale for embedding-based recommendations."""

    def __init__(self, config: SimilarityExplanationConfig | None = None) -> None:
        self.config = config or SimilarityExplanationConfig()

    def explain(self, seed: SeedContext, candidates: list[CandidateContext]) -> str:
        """Return a textual explanation for why candidates match the seed."""
        prompt = self._build_prompt(seed, candidates)
        return self._run(prompt)

    def _run(self, prompt: str) -> str:
        coro = self._sample(prompt)
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:
            if "asyncio.run() cannot be called" not in str(exc):
                raise
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)

    async def _sample(self, prompt: str) -> str:
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY environment variable is required for explanations.")
        client = AsyncClient(api_key=api_key, timeout=self.config.timeout_seconds)
        create_kwargs: dict[str, object] = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.top_p is not None:
            create_kwargs["top_p"] = self.config.top_p

        messages = [
            grok_system(
                "You are an analytical curator explaining why a recommended title surfaces next to a seed title in an embedding search. Keep the tone plainspoken, not florid or critic-like. "
                "Synthesize the supplied analyses to extract the clearest thematic, structural, or tonal throughlines, and articulate them in a precise, even-handed tone. "
                "Summarize the resonance without retelling the recommended plot; focus on motifs, character arcs, pacing, or craft choices that the analyses highlight. "
                "Explicitly note why those overlaps might interest a fan of the seed, and call out at least one substantive difference or caveat drawn from the analyses so the reader can judge the fit or potential mismatch."
            )
        ]
        try:
            chat = client.chat.create(messages=messages, **create_kwargs)
            chat.append(grok_user(prompt))
            response = await chat.sample()
        finally:
            close_method = getattr(client, "close", None)
            if callable(close_method):
                result = close_method()
                if asyncio.iscoroutine(result):
                    await result

        content = getattr(response, "content", "")
        if not isinstance(content, str):
            raise RuntimeError("Unexpected Grok response payload for explanation.")
        return content.strip()

    def _build_prompt(self, seed: SeedContext, candidates: list[CandidateContext]) -> str:
        seed_summary = _truncate(seed.summary, self.config.max_seed_chars)
        lines: list[str] = [
            f"Seed title: {seed.label}",
            "",
            "Seed analysis:",
            seed_summary,
            "",
            "Recommended titles:",
        ]
        for idx, candidate in enumerate(candidates, start=1):
            candidate_summary = _truncate(candidate.summary, self.config.max_candidate_chars)
            lines.extend(
                [
                    f"{idx}. {candidate.label} (similarity score {candidate.score:.4f})",
                    "Analysis:",
                    candidate_summary,
                    "",
                ]
            )
        lines.append(
            "Provide a focused recommendation that covers: "
            "(a) the sharpest shared themes, structural moves, or tonal signatures surfaced in the analyses, "
            "(b) a brief explanation of why those overlaps matter to someone who appreciated the seed title (or why they might not), "
            "and (c) one or two meaningful contrasts or caveats that the analyses raise. "
            "Avoid plot summaries; emphasize analytical insight. "
            "Keep it concise—no more than two tight paragraphs or up to four bullets."
        )
        return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}…"

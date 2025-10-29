"""Embedding generation and similarity search."""

from __future__ import annotations

import contextlib
import gc
import inspect
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from openai import OpenAI

from .manifest import EmbeddingCandidate, Manifest, TitleRecord
from .paths import PATHS, embeddings_dir

LOGGER = logging.getLogger(__name__)


QWEN_MOVIE_EMBED_INSTRUCTION = (
    "Instruct: For movie recommendations, embed this free-text description so the vector captures viewing vibe, emotional momentum, genre anchors, standout craft, and audience appeal.\nQuery: {text}"
)


@dataclass(frozen=True)
class TorchRuntime:
    """Resolved device and dtype information for local Torch models."""

    device: "torch.device"
    compute_dtype: "torch.dtype"
    autocast_dtype: "torch.dtype | None"
    pinned_memory: bool

    @property
    def device_type(self) -> str:
        return getattr(self.device, "type", "cpu")

    @property
    def is_cuda(self) -> bool:
        return self.device_type == "cuda"

    @property
    def is_mps(self) -> bool:
        return self.device_type == "mps"

    @property
    def device_str(self) -> str:
        return str(self.device)


def _resolve_torch_runtime(preferred_device: str | None) -> TorchRuntime:
    """Pick an execution device and dtype optimized for throughput."""
    import torch

    def _auto_device() -> "torch.device":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preferred_device:
        try:
            device = torch.device(preferred_device)
        except (RuntimeError, TypeError, ValueError) as exc:
            LOGGER.warning("Invalid Torch device '%s'; falling back to auto (%s)", preferred_device, exc)
            device = _auto_device()
    else:
        device = _auto_device()

    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    compute_dtype = torch.float32
    autocast_dtype: "torch.dtype | None" = None

    if device.type == "cuda":
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            torch.set_float32_matmul_precision("medium")
        except AttributeError:
            pass
        with contextlib.suppress(AttributeError):
            torch.backends.cuda.matmul.allow_tf32 = True
        with contextlib.suppress(AttributeError):
            torch.backends.cudnn.allow_tf32 = True
        autocast_dtype = compute_dtype
    elif device.type == "mps":
        compute_dtype = torch.float16
        autocast_dtype = compute_dtype

    return TorchRuntime(
        device=device,
        compute_dtype=compute_dtype,
        autocast_dtype=autocast_dtype,
        pinned_memory=device.type == "cuda",
    )


def _apply_instruction(texts: Sequence[str], instruction: str | None) -> list[str]:
    """Format inputs with an optional task instruction."""
    directive = (instruction or "").strip()
    if not directive:
        return [text if isinstance(text, str) else str(text) for text in texts]
    if "{text}" in directive:
        return [directive.format(text=text) for text in texts]
    prefix = directive
    return [f"{prefix}\n{text}" if text else prefix for text in texts]


def _normalize_numpy(matrix) -> "numpy.ndarray":
    """Normalize rows of a numpy matrix without mutating callers."""
    import numpy as np

    vectors = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def is_qwen_embedding_model(model: str | None) -> bool:
    """Return True if the supplied model name refers to a Qwen embedding checkpoint."""
    if not model:
        return False
    name = model.strip().lower()
    return "qwen3-embedding" in name or name.startswith("qwen/")


def default_instruction_for_model(model: str | None) -> str | None:
    """Return a task-specific instruction string when the model benefits from it."""
    if is_qwen_embedding_model(model):
        return QWEN_MOVIE_EMBED_INSTRUCTION
    return None


def resolve_embedding_instruction(
    provider: str | None,
    model: str | None,
    instruction: str | None,
    *,
    for_query: bool,
) -> str | None:
    """Normalize caller-supplied instructions and fill in defaults when needed."""
    value = (instruction or "").strip()
    if value:
        return value
    if not for_query:
        return None
    provider_key = (provider or "").strip().lower()
    if provider_key in {"sentence-transformers", "huggingface", "hf"}:
        return default_instruction_for_model(model)
    return None


def save_embedding_vector(tconst: str, vector: list[float] | "np.ndarray", directory: Path) -> Path:
    """Persist an embedding vector to the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    import numpy as np

    path = directory / f"{tconst}.npy"
    arr = np.asarray(vector, dtype=np.float32)
    np.save(path, arr, allow_pickle=False)
    _mark_embeddings_cache_dirty(directory)
    return path


def _embedding_cache_paths(directory: Path) -> tuple[Path, Path]:
    cache_dir = directory / ".cache"
    matrix_path = cache_dir / "matrix.npz"
    meta_path = cache_dir / "matrix.meta.json"
    return matrix_path, meta_path


def _mark_embeddings_cache_dirty(directory: Path) -> None:
    _, meta_path = _embedding_cache_paths(directory)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "dirty": True}
    meta_path.write_text(json.dumps(payload), encoding="utf-8")


def _snapshot_embedding_directory(files: Sequence[Path]) -> list[dict[str, int | str]]:
    snapshot: list[dict[str, int | str]] = []
    for file in files:
        try:
            stat = file.stat()
        except OSError:
            continue
        snapshot.append(
            {
                "name": file.name,
                "size": int(stat.st_size),
                "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            }
        )
    return snapshot


def _load_cached_matrix(directory: Path, snapshot: list[dict[str, int | str]]):
    matrix_path, meta_path = _embedding_cache_paths(directory)
    if not matrix_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("version") != 1 or meta.get("dirty"):
        return None
    if meta.get("manifest") != snapshot:
        return None
    try:
        import numpy as np

        payload = np.load(matrix_path, allow_pickle=False)
        vectors = payload["vectors"]
        ids = payload["ids"].tolist()
    except Exception:
        return None
    dimension = int(meta.get("dimension") or (vectors.shape[1] if vectors.size else 0))
    return ids, vectors.astype(np.float32, copy=False), dimension


def _write_cached_matrix(
    directory: Path,
    ids: list[str],
    matrix,
    snapshot: list[dict[str, int | str]],
) -> None:
    matrix_path, meta_path = _embedding_cache_paths(directory)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np

        vectors = np.asarray(matrix, dtype=np.float32)
        np.savez(
            matrix_path,
            vectors=vectors,
            ids=np.asarray(ids, dtype=object),
        )
        meta_payload = {
            "version": 1,
            "dirty": False,
            "dimension": int(vectors.shape[1]) if vectors.size else 0,
            "vector_count": len(ids),
            "manifest": snapshot,
        }
        meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")
    except Exception:
        with contextlib.suppress(Exception):
            matrix_path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            meta_path.unlink(missing_ok=True)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding computation."""

    provider: str = "openai"
    model: str = "text-embedding-3-large"
    embedding_set: str = "default"
    batch_size: int = 32
    timeout_seconds: int = 60
    dry_run: bool = False
    dimensions: int | None = 1024
    device: str | None = None
    instruction: str | None = None
    max_length: int | None = 8192


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


class SentenceTransformerEmbeddingClient:
    """Embedding client backed by sentence-transformers models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        import os
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "sentence-transformers is required for provider 'sentence-transformers'. "
                "Install it via `pip install sentence-transformers`."
            ) from exc

        self.config = config
        self.batch_size = max(1, config.batch_size)
        self.runtime = _resolve_torch_runtime(config.device)
        self._instruction = (config.instruction or "").strip()
        os.environ.setdefault("HF_USE_FLASH_ATTENTION", "1")

        init_kwargs: dict[str, object] = {"tokenizer_kwargs": {"padding_side": "left"}}
        model_kwargs: dict[str, object] = {}
        if self.runtime.is_cuda or self.runtime.is_mps:
            model_kwargs["torch_dtype"] = self.runtime.compute_dtype
        if model_kwargs:
            init_kwargs["model_kwargs"] = model_kwargs
        if config.device:
            init_kwargs["device"] = config.device

        try:
            self.model = SentenceTransformer(config.model, **init_kwargs)
        except TypeError:
            # Older sentence-transformers releases may not understand torch_dtype.
            init_kwargs.pop("model_kwargs", None)
            self.model = SentenceTransformer(config.model, **init_kwargs)

        try:
            self.model.eval()
        except AttributeError:
            pass
        with contextlib.suppress(AttributeError):
            self.model.requires_grad_(False)  # type: ignore[attr-defined]
        with contextlib.suppress(Exception):
            self.model.to(self.runtime.device)

        self._auto_normalizes = False
        self._encode_kwargs: dict[str, object] = {
            "batch_size": self.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }

        try:
            encode_params = inspect.signature(self.model.encode).parameters
        except (TypeError, ValueError):
            encode_params = {}

        if "device" in encode_params:
            self._encode_kwargs["device"] = self.runtime.device_str

        if "normalize_embeddings" in encode_params:
            self._encode_kwargs["normalize_embeddings"] = True
            self._auto_normalizes = True

        if "num_workers" in encode_params:
            workers = max(1, min(4, (os.cpu_count() or 1) // 2 or 1))
            self._encode_kwargs["num_workers"] = workers

        if "pin_memory" in encode_params:
            self._encode_kwargs["pin_memory"] = self.runtime.pinned_memory

        if "use_amp" in encode_params and self.runtime.autocast_dtype is not None:
            self._encode_kwargs["use_amp"] = True

        max_length = config.max_length or 8192
        with contextlib.suppress(Exception):
            current = getattr(self.model, "max_seq_length", max_length)
            if current is None or current > max_length:
                self.model.max_seq_length = max_length
        with contextlib.suppress(Exception):
            tokenizer = getattr(self.model, "tokenizer", None)
            if tokenizer is not None and getattr(tokenizer, "model_max_length", None):
                tokenizer.model_max_length = max_length

    def embed(self, texts: list[str]) -> tuple[list[list[float]], dict[str, int]]:
        inputs = _apply_instruction(texts, self._instruction)
        try:
            import torch

            inference_ctx = torch.inference_mode()
        except (ModuleNotFoundError, AttributeError):
            inference_ctx = contextlib.nullcontext()

        with inference_ctx:
            vectors = self.model.encode(inputs, **self._encode_kwargs)
        if not self._auto_normalizes:
            vectors = _normalize_numpy(vectors)
        embeddings = vectors.tolist()
        return embeddings, {"total_tokens": 0, "prompt_tokens": 0}

    def flush(self) -> None:
        if not self.runtime.is_cuda:
            return
        try:
            import torch

            torch.cuda.synchronize(self.runtime.device)
            torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self) -> None:
        try:
            import torch

            if self.runtime.is_cuda:
                torch.cuda.empty_cache()
        except Exception:
            pass
        with contextlib.suppress(Exception):
            self.model.to("cpu")
        with contextlib.suppress(Exception):
            del self.model
        gc.collect()


def _last_token_pool(last_hidden_states, attention_mask, device_type: str) -> "torch.Tensor":
    """Pool embeddings based on the final non-padding token."""
    import torch

    if device_type == "mps":
        # MPS pads on the right; relying on attention mask is safer.
        sequence_lengths = attention_mask.sum(dim=1) - 1
    else:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1

    batch_indices = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths.clamp(min=0)]


class QwenEmbeddingClient:
    """Optimized embedding client for Qwen3 embedding models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import is_flash_attn_2_available

        self.config = config
        self.batch_size = max(1, config.batch_size)
        self.max_length = config.max_length or 8192
        self.runtime = _resolve_torch_runtime(config.device)
        self._instruction = (config.instruction or "").strip()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            padding_side="left",
            truncation_side="left",
        )
        self.tokenizer.model_max_length = self.max_length
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        backend_tokenizer = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend_tokenizer is not None:
            with contextlib.suppress(Exception):
                backend_tokenizer.enable_truncation(max_length=self.max_length)
                backend_tokenizer.enable_padding(
                    direction="left",
                    pad_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0,
                    pad_type_id=0,
                    pad_token=self.tokenizer.pad_token or self.tokenizer.eos_token or self.tokenizer.unk_token,
                )

        model_kwargs: dict[str, object] = {"low_cpu_mem_usage": True}
        if self.runtime.is_cuda or self.runtime.is_mps:
            model_kwargs["torch_dtype"] = self.runtime.compute_dtype
        if self.runtime.is_cuda and is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            self.model = AutoModel.from_pretrained(config.model, **model_kwargs)
        except TypeError:
            fallback_kwargs = dict(model_kwargs)
            dtype_value = fallback_kwargs.pop("torch_dtype", None)
            if dtype_value is not None:
                fallback_kwargs["dtype"] = dtype_value
            self.model = AutoModel.from_pretrained(config.model, **fallback_kwargs)

        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.runtime.device)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self._tokenizer_kwargs: dict[str, object] = {
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if self.runtime.is_cuda:
            self._tokenizer_kwargs["pad_to_multiple_of"] = 8

    def _should_retry_on_oom(self, error: RuntimeError, batch_size: int) -> bool:
        if batch_size <= 1:
            return False
        message = str(error).lower()
        if "out of memory" in message or "cuda error: out of memory" in message:
            return True
        try:
            import torch

            oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
        except Exception:
            return False
        return bool(oom_error and isinstance(error, oom_error))

    def embed(self, texts: list[str]) -> tuple[list[list[float]], dict[str, int]]:
        if not texts:
            return [], {"total_tokens": 0, "prompt_tokens": 0}

        inputs = _apply_instruction(texts, self._instruction)
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("torch is required for Qwen embeddings.") from exc

        results: list["torch.Tensor"] = []
        batch_size = self.batch_size

        while True:
            try:
                for chunk in batched(inputs, batch_size):
                    encoded = self.tokenizer(chunk, **self._tokenizer_kwargs)
                    encoded = {
                        key: value.to(self.runtime.device, non_blocking=self.runtime.is_cuda)
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in encoded.items()
                    }
                    with torch.inference_mode():
                        autocast_ctx = (
                            torch.autocast(self.runtime.device_type, dtype=self.runtime.autocast_dtype)
                            if self.runtime.autocast_dtype is not None
                            else contextlib.nullcontext()
                        )
                        with autocast_ctx:
                            outputs = self.model(**encoded)
                            pooled = _last_token_pool(
                                outputs.last_hidden_state,
                                encoded["attention_mask"],
                                self.runtime.device_type,
                            )
                            normalized = F.normalize(pooled, p=2, dim=-1)
                    results.append(normalized.detach().cpu())
                    del outputs, pooled, normalized
                break
            except RuntimeError as exc:  # pragma: no cover - hardware dependent
                if not self._should_retry_on_oom(exc, batch_size):
                    raise
                previous = batch_size
                batch_size = max(1, batch_size // 2)
                LOGGER.warning(
                    "Qwen embeddings hit OOM; retrying with batch size %d → %d.",
                    previous,
                    batch_size,
                )
                if self.runtime.is_cuda:
                    torch.cuda.empty_cache()
                continue

        if not results:
            return [], {"total_tokens": 0, "prompt_tokens": 0}

        with torch.inference_mode():
            stacked = torch.cat(results, dim=0).to(torch.float32)
        if self.runtime.is_cuda:
            torch.cuda.synchronize(self.runtime.device)
            torch.cuda.empty_cache()
        return stacked.tolist(), {"total_tokens": 0, "prompt_tokens": 0}

    def flush(self) -> None:
        if not self.runtime.is_cuda:
            return
        try:
            import torch

            torch.cuda.synchronize(self.runtime.device)
            torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self) -> None:
        try:
            import torch

            with contextlib.suppress(Exception):
                self.model.to("cpu")
            if self.runtime.is_cuda:
                torch.cuda.empty_cache()
        except Exception:
            pass
        with contextlib.suppress(Exception):
            del self.model
        with contextlib.suppress(Exception):
            del self.tokenizer
        gc.collect()


class EmbeddingRunner:
    """Generate embeddings for Grok analyses and persist them."""

    def __init__(self, manifest: Manifest, config: EmbeddingConfig | None = None) -> None:
        self.manifest = manifest
        self.config = config or EmbeddingConfig()
        profile = getattr(self.manifest, "profile", None)
        if profile is None or not str(profile).strip():
            raise ValueError("Manifest.profile must be set before generating embeddings.")
        self.profile = str(profile).strip()
        self.variant = (self.config.embedding_set or "default").strip() or "default"
        self.embeddings_dir = embeddings_dir(self.profile, self.variant)
        provider_key = (self.config.provider or "openai").strip().lower()
        self.provider = provider_key

        if provider_key == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key and not self.config.dry_run:
                raise RuntimeError("OPENAI_API_KEY environment variable is required.")
            self.client = OpenAIEmbeddingClient(api_key or "dry-run-key", self.config)
        elif provider_key in {"sentence-transformers", "huggingface", "hf"}:
            if is_qwen_embedding_model(self.config.model):
                self.client = QwenEmbeddingClient(self.config)
            else:
                self.client = SentenceTransformerEmbeddingClient(self.config)
            self.provider = "sentence-transformers"
        else:
            raise ValueError(f"Unsupported embedding provider '{self.config.provider}'.")

    def run(self, *, limit: int | None = None, force: bool = False) -> None:
        candidates = list(
            self.manifest.iter_embedding_candidates(
                limit,
                include_completed=force,
                profile=self.profile,
                variant=self.variant,
            )
        )
        if not candidates:
            LOGGER.warning("No embedding candidates found.")
            return

        if force:
            LOGGER.info("Force flag enabled; regenerating embeddings for %d titles.", len(candidates))
            for candidate in candidates:
                self._prepare_for_regeneration(candidate.title.tconst)

        total_titles = len(candidates)
        batch_size = max(1, self.config.batch_size)
        total_batches = (total_titles + batch_size - 1) // batch_size
        LOGGER.info(
            "Generating embeddings for %d analyses (provider=%s model=%s batch=%d)",
            total_titles,
            self.provider,
            self.config.model,
            batch_size,
        )
        session_id = f"emb-{uuid.uuid4().hex[:10]}"
        scope = self.profile if self.variant == "default" else f"{self.profile}/{self.variant}"
        component_name = f"{self.provider}_embeddings[{scope}]"
        self.manifest.register_session(session_id, component=component_name)

        total_tokens = 0
        total_prompt_tokens = 0
        processed = 0
        overall_start = time.perf_counter()
        for batch_index, batch in enumerate(batched(candidates, batch_size), start=1):
            batch_start = time.perf_counter()
            texts = [self._read_analysis(candidate) for candidate in batch]
            ids = [candidate.title.tconst for candidate in batch]
            if self.config.dry_run:
                processed += len(batch)
                elapsed = time.perf_counter() - batch_start
                elapsed_total = time.perf_counter() - overall_start
                rate = processed / elapsed_total if elapsed_total > 0 else 0.0
                LOGGER.info(
                    "Batch %d/%d [dry-run] | %d titles | %.2fs (%.1f/s) | processed=%d/%d",
                    batch_index,
                    total_batches,
                    len(batch),
                    elapsed,
                    rate,
                    processed,
                    total_titles,
                )
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
                        variant=self.variant,
                        provider=self.provider,
                        session_id=session_id,
                        error=str(exc),
                )
                continue

            batch_tokens = usage.get("total_tokens", 0)
            total_tokens += batch_tokens
            total_prompt_tokens += usage.get("prompt_tokens", 0)
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
                        variant=self.variant,
                        provider=self.provider,
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
                    variant=self.variant,
                    provider=self.provider,
                    session_id=session_id,
                    model=self.config.model,
                    dim=len(vector),
                    vector_path=str(vector_path),
                    input_tokens=per_doc_tokens,
                )

            processed += len(batch)
            elapsed = time.perf_counter() - batch_start
            elapsed_total = time.perf_counter() - overall_start
            rate = processed / elapsed_total if elapsed_total > 0 else 0.0
            LOGGER.info(
                "Batch %d/%d | %d titles | %.2fs (%.1f/s) | tokens=%d | processed=%d/%d",
                batch_index,
                total_batches,
                len(batch),
                elapsed,
                rate,
                batch_tokens,
                processed,
                total_titles,
            )

        if hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Failed to flush embedding client cache: %s", exc)

        self.manifest.finalize_session(
            session_id,
            total_input_tokens=total_tokens,
            total_cached_input_tokens=0,
            total_output_tokens=0,
            total_reasoning_tokens=0,
            total_cost=self._estimate_cost(total_tokens),
            notes=(
                f"profile={self.profile} variant={self.variant} provider={self.provider} "
                f"titles={len(candidates)}"
            ),
        )
        elapsed_total = time.perf_counter() - overall_start
        LOGGER.info(
            "Embedding generation complete (%d titles) in %.2fs | tokens=%d prompt_tokens=%d",
            len(candidates),
            elapsed_total,
            total_tokens,
            total_prompt_tokens,
        )

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
            variant=self.variant,
            provider=self.provider,
            session_id=None,
            model=None,
            prompt_hash=None,
            dim=0,
            input_tokens=0,
            vector_path="",
            error=None,
        )

    def _estimate_cost(self, tokens: int) -> float:
        if self.provider != "openai":
            return 0.0
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
    snapshot = _snapshot_embedding_directory(files)
    cached = _load_cached_matrix(directory, snapshot)
    if cached is not None:
        return cached

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
    dimension = dims or (matrix.shape[1] if matrix.size else 0)
    if ids and matrix.size:
        _write_cached_matrix(directory, ids, matrix, snapshot)
    return ids, matrix, dimension


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


def intersection_similar(
    matrix,
    seed_vectors: Sequence[Sequence[float]],
    ids: list[str],
    *,
    top_k: int = 10,
    largest: bool = True,
) -> list[tuple[str, float]]:
    """Return titles ranked by the minimum similarity across all seed vectors."""
    import numpy as np

    if top_k <= 0 or matrix.size == 0:
        return []
    normalized: list[np.ndarray] = []
    for index, vector in enumerate(seed_vectors):
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            LOGGER.warning("Skipping zero-length seed embedding at index %d", index)
            continue
        normalized.append(arr / norm)
    if not normalized:
        raise ValueError("At least one non-zero seed embedding is required.")

    seed_stack = np.stack(normalized, axis=0)  # (num_seeds, dim)
    score_matrix = matrix @ seed_stack.T  # (num_titles, num_seeds)
    aggregated = score_matrix.min(axis=1)

    total = aggregated.shape[0]
    limit = min(top_k, total)
    if limit == 0:
        return []

    if largest:
        partition_scores = -aggregated
        sort_key = lambda item: -item[1]
    else:
        partition_scores = aggregated
        sort_key = lambda item: item[1]

    best_indices = np.argpartition(partition_scores, limit - 1)[:limit]
    best_pairs = ((ids[i], float(aggregated[i])) for i in best_indices)
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

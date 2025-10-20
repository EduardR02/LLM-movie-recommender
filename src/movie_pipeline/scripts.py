from __future__ import annotations

import hashlib
from pathlib import Path

from .manifest import Manifest, TitleRecord


def _load_plot_hash(manifest: Manifest, record: TitleRecord) -> str | None:
    plot = manifest.get_plot_record(record.tconst)
    return plot.plot_hash if plot else None


def import_analyses_from_dir(
    directory: Path,
    *,
    profile: str = "default",
    model: str = "grok-4-fast-reasoning-latest",
) -> int:
    manifest = Manifest(profile=profile)
    count = 0
    for path in directory.glob("*.txt"):
        record = manifest.get_title(path.stem)
        if not record:
            continue
        text = path.read_text(encoding="utf-8")
        hash_on_disk = hashlib.sha256(text.encode("utf-8")).hexdigest()
        plot_hash = _load_plot_hash(manifest, record)
        manifest.ensure_analysis_record(
            record,
            profile=profile,
            plot_hash=plot_hash or hash_on_disk,
            path=path,
            model=model,
        )
        count += 1
    manifest.close()
    return count

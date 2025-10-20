Cheat Sheet

General switches
- `--profile NAME` selects an alternate prompt/embedding set (default: `default`).
- `--limit N` caps how many titles a stage processes.
- `--force` overwrites existing artifacts for the active profile.
- `--compare K` prints the K‑th recommendation’s analysis alongside the query seed.

Pipeline
- Fetch IMDb snapshot: `uv run python -m movie_pipeline.cli fetch-imdb --min-rating 6.0 --min-votes 10000`
- Grab plots: `uv run python -m movie_pipeline.cli fetch-plots`
- Run Grok analyses: `uv run python -m movie_pipeline.cli run-analysis --max-concurrency 8`
- Alternate prompt (e.g., profile `different`, up to 1.5k titles): `uv run python -m movie_pipeline.cli --profile different run-analysis --limit 1500 --max-concurrency 100`
- Embed analyses with OpenAI: `uv run python -m movie_pipeline.cli compute-embeddings --dimensions 1024 --batch-size 64`
- Embed alternate profile: `uv run python -m movie_pipeline.cli --profile different compute-embeddings --dimensions 1024 --batch-size 64`

Querying & inspection
- Free-text neighbors (analysis embeddings): `uv run python -m movie_pipeline.cli query-text "search string" --top-k 10 --index analysis`
- Title neighbors: `uv run python -m movie_pipeline.cli query-title "Breaking Bad" --top-k 10 --index analysis`
- Limit output to a profile & peek at reasoning: add `--profile NAME --compare 2 --explain --explain-top 3`
- Show stored artifacts: `uv run python -m movie_pipeline.cli show-title "Breaking Bad" --plot --analysis --lines 40`

Operational
- Session summary: `uv run python -m movie_pipeline.cli sessions --aggregate`
- Clear analyses for a title: `uv run python -m movie_pipeline.cli clear-analyses --identifier "Breaking Bad"`
- List manifest titles: `uv run python -m movie_pipeline.cli list-titles`

Other commands exist for plot embeddings, raw plot searches, and quick 2D embedding scatters (`python -m movie_pipeline.embedding_plot --profile NAME`). Use the same patterns with `compute-plot-embeddings` or `--index plot` when needed.

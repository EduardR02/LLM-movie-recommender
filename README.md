# Movie Recommending Pipeline

Instead of doing the usual wikipedia plot summary -> embeddings -> cosine similarity, we do:

- filter movies and tv series first for convenience, currently > 6.0 imdb rating and > 10000 votes (this is 11.7k movies and tv shows as of today)
- get their wikipedia plot summaries and descriptions
- use grok-4-fast-reasoning and have it TRY ITS BEST to generate why a movie is compelling based on the wikipedia data only as a starting point to keep hallucination minimal
- this is important - wikipedia plot summaries are not perfect for why a movie is compelling to watch - there are many other things. Take Blade Runner 2049 as an example. The plot summary is almost useless for why this movie is cool. this is why we need grok - it adds all these additional aspects for why one might like a movie
- we run this for all the 11.7k movies and tv shows - lol.
- then we take the grok texts and embed them using openai's text-embedding-3-large (reduced to 1024 dim because that doesn't seem to affect performance)

and boom - hopefully we now have a better recommender than the naive version.

## Thoughts

How well the recommendation system works is likely highly dependent on the quality of the grok texts. so better prompt - better embeddings.
The system prompt can probably be refined a lot and engineered by how it affects the output for a certain movie after each refinement.
To give you an idea of how the current output looks (for system_different.txt), look at example_grok_output.txt - this is for Blade Runner 2049.

Running grok through 11.7k movies is crazy. Intelligence is so cheap now that you can just brute force. One pass costs around $11. lol. you can even get this further down if you try to make it less verbose, but im not sure how that will affect recommendation quality - it could even go up!

Also funny, this entire project was writtien by codex-cli with gpt-5-codex - idea to final version was maybe a day, but actual implementation maybe a few hours.



## Highlights
- **Ingestion** – `fetch-imdb` downloads the official IMDb TSV snapshots and writes a filtered manifest into SQLite.
- **Plot harvesting** – `fetch-plots` maps titles to Wikipedia pages, cleans the text, and stores both raw and cleaned plots.
- **LLM enrichment** – `run-analysis` calls Grok (xAI) to produce rich per-title analyses, tracking usage and status per profile.
- **Embeddings** – `compute-embeddings` (and `compute-plot-embeddings`) send batches to OpenAI for vector representations, saved per profile for A/B prompts.
- **Queries** – `query-text` and `query-title` surface nearest neighbours with optional analysis snippets and Grok-on-Grok explanations.
- **Visualization** – `python -m movie_pipeline.embedding_plot` renders quick 2D scatters (ratings/genre tinted) for sanity checks.

Refer to `cheat_sheet.md` for CLI examples and workflow switches (profiles, limits, force rebuilds). The `prompts/` directory contains the system prompts used during Grok runs so each profile can be versioned alongside the code.

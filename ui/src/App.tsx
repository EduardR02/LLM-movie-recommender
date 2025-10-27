import { type FormEvent, useEffect, useMemo, useState } from 'react';
import './App.css';

type TitleSummary = {
  tconst: string;
  primary_title: string;
  original_title: string | null;
  title_type: string;
  start_year: number | null;
  end_year: number | null;
  runtime_minutes: number | null;
  genres: string | null;
  num_votes: number;
  average_rating: number;
  sort_rank: number;
};

type Recommendation = {
  title: TitleSummary;
  score: number;
  analysis_preview?: string | null;
  plot_preview?: string | null;
};

type RecommendationResponse = {
  seeds: TitleSummary[];
  results: Recommendation[];
};

type TitleDetailResponse = {
  title: TitleSummary;
  plot?: string | null;
  analysis?: string | null;
  suggestions?: TitleSummary[] | null;
};

type TextMatch = {
  tconst: string;
  primary_title?: string | null;
  start_year?: number | null;
  snippet: string;
};

type ToolKey = 'title' | 'text' | 'library' | 'needle';

type ExpansionEntry = {
  analysis?: string | null;
  plot?: string | null;
  explanation?: string | null;
  explanationError?: string | null;
  showExplanation?: boolean;
  loadingAnalysis?: boolean;
  loadingPlot?: boolean;
  loadingExplanation?: boolean;
};

const DEFAULT_SEEDS = ['Blade Runner 2049', 'Her'].join('\n');

const clampTopKInput = (rawValue: string, fallback: number) => {
  const trimmed = rawValue.trim();
  if (!trimmed) {
    return fallback;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(100, Math.max(1, Math.round(parsed)));
};

const TOOL_META: Record<
  ToolKey,
  {
    label: string;
    eyebrow: string;
    hint: string;
    cta?: string;
    panelHint?: string | null;
  }
> = {
  title: {
    label: 'Title search',
    eyebrow: 'title mode',
    hint: 'Start with one favorite; add extra lines only when you really want to blend vibes.',
    panelHint: null,
    cta: 'Click any result to jump into its neighborhood.',
  },
  text: {
    label: 'Text vibe search',
    eyebrow: 'text mode',
    hint: 'Describe the mood and let the embeddings find a match.',
  },
  library: {
    label: 'Manifest explorer',
    eyebrow: 'manifest',
    hint: 'Browse IMDb metadata plus stored Grok analyses/plots.',
  },
  needle: {
    label: 'Raw text search',
    eyebrow: 'needle hunter',
    hint: 'Find exact phrases inside Grok outputs or cleaned plots.',
  },
};

const formatTitle = (title: TitleSummary) => {
  const year = title.start_year ?? '????';
  return `${title.primary_title} (${year})`;
};

const getImdbUrl = (tconst: string) => `https://www.imdb.com/title/${tconst}/`;

const numberFormatter = new Intl.NumberFormat('en-US');

function App() {
  const [profiles, setProfiles] = useState<string[]>(['default']);
  const [indexes, setIndexes] = useState<string[]>(['analysis']);
  const [selectedProfile, setSelectedProfile] = useState('default');
  const [selectedIndex, setSelectedIndex] = useState('analysis');
  const [activeTool, setActiveTool] = useState<ToolKey>('title');

  useEffect(() => {
    const loadInitial = async () => {
      try {
        const profilePayload = await fetch('/api/profiles').then((res) => res.json());
        const indexPayload = await fetch('/api/indexes').then((res) => res.json());
        if (Array.isArray(profilePayload?.profiles) && profilePayload.profiles.length) {
          setProfiles(profilePayload.profiles);
          if (!profilePayload.profiles.includes(selectedProfile)) {
            setSelectedProfile(profilePayload.profiles[0]);
          }
        }
        if (Array.isArray(indexPayload?.indexes) && indexPayload.indexes.length) {
          setIndexes(indexPayload.indexes);
          if (!indexPayload.indexes.includes(selectedIndex)) {
            setSelectedIndex(indexPayload.indexes[0]);
          }
        }
      } catch (error) {
        console.error('Failed to bootstrap UI', error);
      }
    };
    loadInitial();
  }, []);

  const [titleSeeds, setTitleSeeds] = useState(DEFAULT_SEEDS);
  const [titleTopK, setTitleTopK] = useState(10);
  const [titleTopKInput, setTitleTopKInput] = useState('10');
  const [titleScoreMode, setTitleScoreMode] =
    useState<'centroid' | 'intersection'>('centroid');
  const [titleLeastSimilar, setTitleLeastSimilar] = useState(false);
  const [includeAnalysis, setIncludeAnalysis] = useState(true);
  const [includePlot, setIncludePlot] = useState(false);
  const [titleControlsCollapsed, setTitleControlsCollapsed] = useState(false);
  const [titleControlsPinned, setTitleControlsPinned] = useState(false);

  const parsedSeeds = useMemo(
    () =>
      titleSeeds
        .split('\n')
        .map((seed) => seed.trim())
        .filter(Boolean),
    [titleSeeds],
  );

  const [titleQueryLoading, setTitleQueryLoading] = useState(false);
  const [titleQueryError, setTitleQueryError] = useState<string | null>(null);
  const [titleResponse, setTitleResponse] =
    useState<RecommendationResponse | null>(null);

  const [textQuery, setTextQuery] = useState(
    'slow-burn technoir mystery with synth score',
  );
  const [textTopK, setTextTopK] = useState(10);
  const [textTopKInput, setTextTopKInput] = useState('10');
  const [textIncludeAnalysis, setTextIncludeAnalysis] = useState(true);
  const [textIncludePlot, setTextIncludePlot] = useState(false);
  const [textLoading, setTextLoading] = useState(false);
  const [textError, setTextError] = useState<string | null>(null);
  const [textResponse, setTextResponse] =
    useState<RecommendationResponse | null>(null);

  const [titleSearchTerm, setTitleSearchTerm] = useState('');
  const [titleMatches, setTitleMatches] = useState<TitleSummary[]>([]);
  const [titleSearchLoading, setTitleSearchLoading] = useState(false);
  const [titleSearchError, setTitleSearchError] = useState<string | null>(null);

  const [detailIncludePlot, setDetailIncludePlot] = useState(true);
  const [detailIncludeAnalysis, setDetailIncludeAnalysis] = useState(true);
  const [selectedTitle, setSelectedTitle] = useState<TitleDetailResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const [expansions, setExpansions] = useState<Record<string, ExpansionEntry>>({});

  const [textSearchTerm, setTextSearchTerm] = useState('');
  const [textSearchTarget, setTextSearchTarget] =
    useState<'analysis' | 'plot'>('analysis');
  const [textMatches, setTextMatches] = useState<TextMatch[]>([]);
  const [textSearchLoading, setTextSearchLoading] = useState(false);
  const [textSearchError, setTextSearchError] = useState<string | null>(null);

  const runTitleQuery = async (identifiers: string[]) => {
    if (!identifiers.length) {
      setTitleQueryError('Add at least one seed title or IMDb ID.');
      return;
    }
    const resolvedTopK = clampTopKInput(titleTopKInput, titleTopK);
    if (resolvedTopK !== titleTopK) {
      setTitleTopK(resolvedTopK);
    }
    if (titleTopKInput !== String(resolvedTopK)) {
      setTitleTopKInput(String(resolvedTopK));
    }
    setTitleQueryLoading(true);
    setTitleQueryError(null);
    try {
      const response = await fetch('/api/query/title', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile: selectedProfile,
          index: selectedIndex,
          identifiers,
          top_k: resolvedTopK,
          score_mode: titleScoreMode,
          least_similar: titleLeastSimilar,
          include_analysis: includeAnalysis,
          include_plot: includePlot,
        }),
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Title query failed');
      }
      const payload: RecommendationResponse = await response.json();
      setTitleResponse(payload);
      if (!titleControlsPinned) {
        setTitleControlsCollapsed(true);
      }
    } catch (error) {
      console.error(error);
      setTitleQueryError(
        error instanceof Error ? error.message : 'Title query failed',
      );
    } finally {
      setTitleQueryLoading(false);
    }
  };

  const handleTitleQuery = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await runTitleQuery(parsedSeeds);
  };

  const handleTextQuery = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!textQuery.trim()) {
      setTextError('Enter a phrase describing what you want to watch.');
      return;
    }
    const resolvedTopK = clampTopKInput(textTopKInput, textTopK);
    if (resolvedTopK !== textTopK) {
      setTextTopK(resolvedTopK);
    }
    if (textTopKInput !== String(resolvedTopK)) {
      setTextTopKInput(String(resolvedTopK));
    }
    setTextLoading(true);
    setTextError(null);
    try {
      const response = await fetch('/api/query/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile: selectedProfile,
          index: selectedIndex,
          text: textQuery,
          top_k: resolvedTopK,
          include_analysis: textIncludeAnalysis,
          include_plot: textIncludePlot,
        }),
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Text query failed');
      }
      const payload: RecommendationResponse = await response.json();
      setTextResponse(payload);
    } catch (error) {
      console.error(error);
      setTextError(error instanceof Error ? error.message : 'Text query failed');
    } finally {
      setTextLoading(false);
    }
  };

  const runTitleSearch = async () => {
    if (!titleSearchTerm.trim()) {
      setTitleMatches([]);
      setTitleSearchError('Enter a partial title.');
      return;
    }
    setTitleSearchLoading(true);
    setTitleSearchError(null);
    try {
      const params = new URLSearchParams({
        q: titleSearchTerm.trim(),
        profile: selectedProfile,
        limit: '12',
      });
      const payload = await fetch(`/api/titles/search?${params.toString()}`).then(
        (res) => res.json(),
      );
      setTitleMatches(payload.results ?? []);
      setSelectedTitle(null);
    } catch (error) {
      console.error(error);
      setTitleSearchError(
        error instanceof Error ? error.message : 'Search failed',
      );
    } finally {
      setTitleSearchLoading(false);
    }
  };

  const handleLibrarySearch = async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();
    await runTitleSearch();
  };

  const fetchFullText = async (tconst: string, { analysis, plot }: { analysis?: boolean; plot?: boolean }) => {
    if (!analysis && !plot) {
      return;
    }
    setExpansions((prev) => ({
      ...prev,
      [tconst]: {
        ...prev[tconst],
        ...(analysis ? { loadingAnalysis: true } : {}),
        ...(plot ? { loadingPlot: true } : {}),
      },
    }));
    try {
      const params = new URLSearchParams({
        profile: selectedProfile,
        include_analysis: String(Boolean(analysis)),
        include_plot: String(Boolean(plot)),
      });
      const payload: TitleDetailResponse = await fetch(
        `/api/titles/${encodeURIComponent(tconst)}?${params.toString()}`,
      ).then((res) => {
        if (!res.ok) {
          throw new Error('Failed to load text.');
        }
        return res.json();
      });
      setExpansions((prev) => ({
        ...prev,
        [tconst]: {
          ...prev[tconst],
          ...(analysis ? { analysis: payload.analysis } : {}),
          ...(plot ? { plot: payload.plot } : {}),
        },
      }));
    } catch (error) {
      console.error(error);
    } finally {
      setExpansions((prev) => ({
        ...prev,
        [tconst]: {
          ...prev[tconst],
          ...(analysis ? { loadingAnalysis: false } : {}),
          ...(plot ? { loadingPlot: false } : {}),
        },
      }));
    }
  };

  const explainRecommendation = async (rec: Recommendation) => {
    const tconst = rec.title.tconst;
    if (!titleResponse?.seeds?.length) {
      setExpansions((prev) => ({
        ...prev,
        [tconst]: {
          ...prev[tconst],
          explanationError: 'Run a title search first to get context.',
        },
      }));
      return;
    }
    setExpansions((prev) => ({
      ...prev,
      [tconst]: {
        ...prev[tconst],
        loadingExplanation: true,
        explanationError: null,
      },
    }));
    try {
      const response = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile: selectedProfile,
          seed_ids: titleResponse.seeds.map((seed) => seed.tconst),
          candidate_id: tconst,
          candidate_score: rec.score,
        }),
      });
      if (!response.ok) {
        let message = 'Failed to load explanation.';
        try {
          const payload = await response.json();
          if (typeof payload?.detail === 'string') {
            message = payload.detail;
          }
        } catch {
          const text = await response.text();
          if (text) {
            message = text;
          }
        }
        throw new Error(message);
      }
      const payload: { explanation: string } = await response.json();
      setExpansions((prev) => ({
        ...prev,
        [tconst]: {
          ...prev[tconst],
          explanation: payload.explanation,
          explanationError: null,
          showExplanation: true,
        },
      }));
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to load explanation.';
      setExpansions((prev) => ({
        ...prev,
        [tconst]: {
          ...prev[tconst],
          explanationError: message,
        },
      }));
    } finally {
      setExpansions((prev) => {
        const current = prev[tconst];
        if (!current) {
          return prev;
        }
        const nextEntry: ExpansionEntry = { ...current };
        delete nextEntry.loadingExplanation;
        return { ...prev, [tconst]: nextEntry };
      });
    }
  };

  const showStoredExplanation = (tconst: string) => {
    setExpansions((prev) => {
      const current = prev[tconst];
      if (!current) {
        return prev;
      }
      return {
        ...prev,
        [tconst]: {
          ...current,
          showExplanation: true,
        },
      };
    });
  };

  const collapseExpansion = (tconst: string, key: 'analysis' | 'plot' | 'explanation') => {
    setExpansions((prev) => {
      const current = prev[tconst];
      if (!current) {
        return prev;
      }
      const nextEntry: typeof current = { ...current };
      if (key === 'analysis') {
        delete nextEntry.analysis;
        delete nextEntry.loadingAnalysis;
      } else if (key === 'plot') {
        delete nextEntry.plot;
        delete nextEntry.loadingPlot;
      } else {
        nextEntry.showExplanation = false;
      }
      if (!nextEntry.analysis && !nextEntry.plot && !nextEntry.explanation) {
        const { [tconst]: _, ...rest } = prev;
        return rest;
      }
      return { ...prev, [tconst]: nextEntry };
    });
  };

  const jumpToTitle = async (title: TitleSummary) => {
    setActiveTool('title');
    setTitleSeeds(title.tconst);
    await runTitleQuery([title.tconst]);
  };

  const loadDetails = async (identifier: string) => {
    setDetailLoading(true);
    try {
      const params = new URLSearchParams({
        profile: selectedProfile,
        include_plot: String(detailIncludePlot),
        include_analysis: String(detailIncludeAnalysis),
      });
      const payload: TitleDetailResponse = await fetch(
        `/api/titles/${encodeURIComponent(identifier)}?${params.toString()}`,
      ).then((res) => {
        if (!res.ok) {
          throw new Error('Failed to load title details');
        }
        return res.json();
      });
      setSelectedTitle(payload);
    } catch (error) {
      console.error(error);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleTextSearch = async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();
    if (!textSearchTerm.trim()) {
      setTextMatches([]);
      setTextSearchError('Enter a fragment to search for.');
      return;
    }
    setTextSearchLoading(true);
    setTextSearchError(null);
    try {
      const params = new URLSearchParams({
        q: textSearchTerm.trim(),
        target: textSearchTarget,
        profile: selectedProfile,
        limit: '15',
      });
      const payload = await fetch(`/api/text/search?${params.toString()}`).then(
        (res) => {
          if (!res.ok) {
            throw new Error('Text search failed.');
          }
          return res.json();
        },
      );
      setTextMatches(payload.matches ?? []);
    } catch (error) {
      console.error(error);
      setTextSearchError(
        error instanceof Error ? error.message : 'Text search failed',
      );
    } finally {
      setTextSearchLoading(false);
    }
  };

  const toggleTitleControls = () => {
    const next = !titleControlsCollapsed;
    if (!next) {
      setTitleControlsPinned(true);
    }
    setTitleControlsCollapsed(next);
  };

  const renderActivePanel = () => {
    switch (activeTool) {
      case 'title':
        return (
          <div className={`tool-layout ${titleControlsCollapsed ? 'is-collapsed' : ''}`}>
            {!titleControlsCollapsed && (
              <form className="tool-controls" onSubmit={handleTitleQuery}>
                <div className="title-grid">
                  <label className="field seed-field">
                    <span>Seed titles or IMDb IDs (one per line)</span>
                    <textarea
                      value={titleSeeds}
                      onChange={(event) => setTitleSeeds(event.target.value)}
                      rows={5}
                    />
                    <p className="help-text">
                      Start with a single title for straight recs. Add extra lines only when you need a blended tone.
                    </p>
                  </label>
                  <div className="control-stack">
                    <div className="field-row compact">
                      <label className="field">
                        <span>Top K</span>
                        <input
                          type="number"
                          inputMode="numeric"
                          value={titleTopKInput}
                          min={1}
                          max={100}
                          onChange={(event) => setTitleTopKInput(event.target.value)}
                        />
                      </label>
                      <label className="field">
                        <span>Score mode</span>
                        <select
                          value={titleScoreMode}
                          onChange={(event) =>
                            setTitleScoreMode(
                              event.target.value as 'centroid' | 'intersection',
                            )
                          }
                        >
                          <option value="centroid">Centroid (blend)</option>
                          <option value="intersection">Intersection (all seeds)</option>
                        </select>
                      </label>
                    </div>
                    <div className="field-row toggles wrap">
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={titleLeastSimilar}
                          onChange={(event) => setTitleLeastSimilar(event.target.checked)}
                        />
                        <span>Least similar</span>
                      </label>
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={includeAnalysis}
                          onChange={(event) => setIncludeAnalysis(event.target.checked)}
                        />
                        <span>Show Grok preview</span>
                      </label>
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={includePlot}
                          onChange={(event) => setIncludePlot(event.target.checked)}
                        />
                        <span>Show plot preview</span>
                      </label>
                    </div>
                    <div className="action-row">
                      <button
                        type="submit"
                        className="primary"
                        disabled={titleQueryLoading}
                      >
                        {titleQueryLoading ? 'Searching…' : 'Run title search'}
                      </button>
                    </div>
                  </div>
                </div>
              </form>
            )}
            <div className="tool-results-panel">
              <div className="tool-results-header">
                <div>
                  <p className="eyebrow small">Results</p>
                  <h3>Title recommendations</h3>
                </div>
                <button
                  type="button"
                  className="ghost"
                  onClick={toggleTitleControls}
                >
                  {titleControlsCollapsed ? 'Show settings' : 'Hide settings'}
                </button>
              </div>
              {titleQueryError && <p className="error small">{titleQueryError}</p>}
              {titleResponse ? (
                <div className="results result-scroll">
                  <div className="chips">
                    {titleResponse.seeds.map((seed) => (
                      <span key={seed.tconst} className="chip">
                        {formatTitle(seed)}
                      </span>
                    ))}
                  </div>
                  <ul className="result-list">
                    {titleResponse.results.map((rec) => {
                      const expansion = expansions[rec.title.tconst] || {};
                      const analysisText = expansion.analysis ?? rec.analysis_preview;
                      const plotText = expansion.plot ?? rec.plot_preview;
                      const explanationVisible = expansion.showExplanation !== false;
                      return (
                        <li key={rec.title.tconst}>
                          <div className="result-header">
                            <div>
                              <p className="result-title">
                                <a
                                  className="result-title-link"
                                  href={getImdbUrl(rec.title.tconst)}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  {formatTitle(rec.title)}
                                </a>
                              </p>
                              <p className="subdued small">
                                IMDb {rec.title.average_rating.toFixed(1)} ·{' '}
                                {numberFormatter.format(rec.title.num_votes)} votes
                              </p>
                            </div>
                            <div className="result-actions">
                              <button
                                type="button"
                                className="ghost"
                                onClick={() => jumpToTitle(rec.title)}
                                disabled={titleQueryLoading}
                              >
                                Explore
                              </button>
                              <button
                                type="button"
                                className="ghost"
                                onClick={() => {
                                  if (expansion.explanation) {
                                    if (explanationVisible) {
                                      collapseExpansion(rec.title.tconst, 'explanation');
                                    } else {
                                      showStoredExplanation(rec.title.tconst);
                                    }
                                  } else {
                                    void explainRecommendation(rec);
                                  }
                                }}
                                disabled={Boolean(expansion.loadingExplanation)}
                              >
                                {expansion.loadingExplanation
                                  ? 'Explaining…'
                                  : expansion.explanation
                                    ? explanationVisible
                                      ? 'Hide explainer'
                                      : 'Show explainer'
                                    : 'Explain'}
                              </button>
                              <span className="score">{rec.score.toFixed(4)}</span>
                            </div>
                          </div>
                          {analysisText && (
                            <div className="preview-block">
                              <p className="preview">{analysisText}</p>
                              <div className="preview-actions">
                                {expansion.analysis ? (
                                  <button
                                    type="button"
                                    className="ghost"
                                    onClick={() => collapseExpansion(rec.title.tconst, 'analysis')}
                                  >
                                    Collapse analysis
                                  </button>
                                ) : (
                                  <button
                                    type="button"
                                    className="ghost"
                                    onClick={() => fetchFullText(rec.title.tconst, { analysis: true })}
                                    disabled={Boolean(expansion.loadingAnalysis)}
                                  >
                                    {expansion.loadingAnalysis ? 'Loading…' : 'Expand analysis'}
                                  </button>
                                )}
                              </div>
                            </div>
                          )}
                          {plotText && (
                            <div className="preview-block">
                              <p className="preview muted">{plotText}</p>
                              <div className="preview-actions">
                                {expansion.plot ? (
                                  <button
                                    type="button"
                                    className="ghost"
                                    onClick={() => collapseExpansion(rec.title.tconst, 'plot')}
                                  >
                                    Collapse plot
                                  </button>
                                ) : (
                                  <button
                                    type="button"
                                    className="ghost"
                                    onClick={() => fetchFullText(rec.title.tconst, { plot: true })}
                                    disabled={Boolean(expansion.loadingPlot)}
                                  >
                                    {expansion.loadingPlot ? 'Loading…' : 'Expand plot'}
                                  </button>
                                )}
                              </div>
                            </div>
                          )}
                          {expansion.explanation && explanationVisible && (
                            <div className="preview-block">
                              <p className="preview">{expansion.explanation}</p>
                              <div className="preview-actions">
                                <button
                                  type="button"
                                  className="ghost"
                                  onClick={() => collapseExpansion(rec.title.tconst, 'explanation')}
                                >
                                  Collapse explanation
                                </button>
                              </div>
                            </div>
                          )}
                          {expansion.explanationError && (
                            <p className="error small">{expansion.explanationError}</p>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ) : (
                <div className="tool-placeholder">
                  <p>Paste a single IMDb title to anchor the vibe.</p>
                  <p className="subdued small">Add more lines only when you need a custom blend.</p>
                </div>
              )}
            </div>
          </div>
        );
      case 'text':
        return (
          <div className="tool-layout">
            <form className="tool-controls" onSubmit={handleTextQuery}>
              <label className="field">
                <span>Describe the vibe</span>
                <textarea
                  rows={5}
                  value={textQuery}
                  onChange={(event) => setTextQuery(event.target.value)}
                />
                <p className="help-text">Use natural language — we embed the prompt directly.</p>
              </label>
              <div className="control-stack">
                <div className="field-row compact">
                  <label className="field">
                    <span>Top K</span>
                    <input
                      type="number"
                      min={1}
                      max={100}
                      inputMode="numeric"
                      value={textTopKInput}
                      onChange={(event) => setTextTopKInput(event.target.value)}
                    />
                  </label>
                </div>
                <div className="field-row toggles wrap">
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={textIncludeAnalysis}
                      onChange={(event) => setTextIncludeAnalysis(event.target.checked)}
                    />
                    <span>Include Grok snippet</span>
                  </label>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={textIncludePlot}
                      onChange={(event) => setTextIncludePlot(event.target.checked)}
                    />
                    <span>Include plot snippet</span>
                  </label>
                </div>
                <div className="action-row">
                  <button type="submit" className="primary" disabled={textLoading}>
                    {textLoading ? 'Embedding…' : 'Run text search'}
                  </button>
                  {textError && <p className="error">{textError}</p>}
                </div>
              </div>
            </form>
            <div className="tool-results-panel">
              <div className="tool-results-header">
                <div>
                  <p className="eyebrow small">Results</p>
                  <h3>Text recommendations</h3>
                </div>
              </div>
              {textResponse ? (
                <div className="results result-scroll">
                  <p className="seed-label">
                    {textResponse.seeds[0]?.primary_title ?? 'Free-text query'}
                  </p>
                  <ul className="result-list">
                    {textResponse.results.map((rec) => {
                      const expansion = expansions[rec.title.tconst] || {};
                      const analysisText = expansion.analysis ?? rec.analysis_preview;
                      const plotText = expansion.plot ?? rec.plot_preview;
                      return (
                        <li key={rec.title.tconst}>
                          <div className="result-header">
                            <div>
                              <p className="result-title">
                                <a
                                  className="result-title-link"
                                  href={getImdbUrl(rec.title.tconst)}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  {formatTitle(rec.title)}
                                </a>
                              </p>
                              <p className="subdued small">
                                IMDb {rec.title.average_rating.toFixed(1)} ·{' '}
                                {numberFormatter.format(rec.title.num_votes)} votes
                              </p>
                            </div>
                            <div className="result-actions">
                              <button
                                type="button"
                                className="ghost"
                                onClick={() => jumpToTitle(rec.title)}
                                disabled={titleQueryLoading}
                              >
                                Explore
                              </button>
                              <span className="score">{rec.score.toFixed(4)}</span>
                            </div>
                          </div>
                          {analysisText && (
                            <div className="preview-block">
                              <p className="preview">{analysisText}</p>
                              <div className="preview-actions">
                                {expansion.analysis ? (
                                  <button
                                    type="button"
                                    className="ghost"
                                    onClick={() =>
                                      collapseExpansion(rec.title.tconst, 'analysis')
                                    }
                                  >
                                    Collapse analysis
                                  </button>
                                ) : (
                                  <button
                                    type="button"
                                    className="ghost"
                                    onClick={() =>
                                      fetchFullText(rec.title.tconst, { analysis: true })
                                    }
                                    disabled={Boolean(expansion.loadingAnalysis)}
                                  >
                                    {expansion.loadingAnalysis ? 'Loading…' : 'Expand analysis'}
                                  </button>
                                )}
                              </div>
                            </div>
                          )}
                          {plotText && (
                            <div className="preview-block">
                              <p className="preview muted">{plotText}</p>
                              <div className="preview-actions">
                                {expansion.plot ? (
                                  <button
                                    type="button"
                                    className="ghost"
                                    onClick={() => collapseExpansion(rec.title.tconst, 'plot')}
                                  >
                                    Collapse plot
                                  </button>
                                ) : (
                                  <button
                                    type="button"
                                    className="ghost"
                                    onClick={() =>
                                      fetchFullText(rec.title.tconst, { plot: true })
                                    }
                                    disabled={Boolean(expansion.loadingPlot)}
                                  >
                                    {expansion.loadingPlot ? 'Loading…' : 'Expand plot'}
                                  </button>
                                )}
                              </div>
                            </div>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ) : (
                <div className="tool-placeholder">
                  <p>Describe the mood you’re hunting for.</p>
                  <p className="subdued small">We’ll embed it and return a ranked stack.</p>
                </div>
              )}
            </div>
          </div>
        );
      case 'library':
        return (
          <div className="tool-layout">
            <form className="tool-controls" onSubmit={handleLibrarySearch}>
              <label className="field">
                <span>Search IMDb titles</span>
                <input
                  type="text"
                  placeholder="e.g. Strange Days"
                  value={titleSearchTerm}
                  onChange={(event) => setTitleSearchTerm(event.target.value)}
                />
                <p className="help-text">Find titles we already indexed plus saved Grok text.</p>
              </label>
              <div className="action-row">
                <button
                  type="submit"
                  className="primary"
                  disabled={titleSearchLoading}
                >
                  {titleSearchLoading ? 'Searching…' : 'Search manifest'}
                </button>
                {titleSearchError && <p className="error">{titleSearchError}</p>}
              </div>
            </form>
            <div className="tool-results-panel">
              <div className="tool-results-header">
                <div>
                  <p className="eyebrow small">Manifest</p>
                  <h3>Stored metadata</h3>
                </div>
                <div className="detail-controls">
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={detailIncludeAnalysis}
                      onChange={(event) => setDetailIncludeAnalysis(event.target.checked)}
                    />
                    <span>Include Grok analysis</span>
                  </label>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={detailIncludePlot}
                      onChange={(event) => setDetailIncludePlot(event.target.checked)}
                    />
                    <span>Include plot</span>
                  </label>
                  <button
                    type="button"
                    className="secondary"
                    onClick={() =>
                      selectedTitle && loadDetails(selectedTitle.title.tconst)
                    }
                    disabled={!selectedTitle || detailLoading}
                  >
                    {detailLoading ? 'Refreshing…' : 'Refresh'}
                  </button>
                </div>
              </div>
              <div className="library-grid">
                <div className="library-list scrollable">
                  {titleMatches.length === 0 ? (
                    <p className="subdued small">No matches yet. Run a search.</p>
                  ) : (
                    <ul className="library-match-list">
                      {titleMatches.map((match) => (
                        <li key={match.tconst}>
                          <button
                            type="button"
                            onClick={() => loadDetails(match.tconst)}
                            className="library-link"
                          >
                            <strong>{formatTitle(match)}</strong>
                            <span className="subdued small">
                              {match.title_type} · IMDb {match.average_rating.toFixed(1)} ·{' '}
                              {numberFormatter.format(match.num_votes)} votes
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                <div className="library-detail">
                  <div className="detail-scroll">
                    {selectedTitle ? (
                      <>
                        <h3>{formatTitle(selectedTitle.title)}</h3>
                        <p className="subdued small">
                          IMDb {selectedTitle.title.average_rating.toFixed(1)} ·{' '}
                          {numberFormatter.format(selectedTitle.title.num_votes)} votes ·{' '}
                          {selectedTitle.title.genres ?? 'No genres'}
                        </p>
                        {selectedTitle.analysis && (
                          <>
                            <h4>Grok analysis</h4>
                            <pre>{selectedTitle.analysis}</pre>
                          </>
                        )}
                        {selectedTitle.plot && (
                          <>
                            <h4>Plot</h4>
                            <pre>{selectedTitle.plot}</pre>
                          </>
                        )}
                        {!selectedTitle.analysis && !selectedTitle.plot && (
                          <p className="subdued small">No stored text for this title yet.</p>
                        )}
                      </>
                    ) : (
                      <p className="subdued">Select a title to load stored text.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      case 'needle':
      default:
        return (
          <div className="tool-layout">
            <form className="tool-controls" onSubmit={handleTextSearch}>
              <div className="field-row">
                <label className="field">
                  <span>Target</span>
                  <select
                    value={textSearchTarget}
                    onChange={(event) =>
                      setTextSearchTarget(event.target.value as 'analysis' | 'plot')
                    }
                  >
                    <option value="analysis">Analyses</option>
                    <option value="plot">Plots</option>
                  </select>
                </label>
              </div>
              <label className="field">
                <span>Contains phrase</span>
                <input
                  type="text"
                  value={textSearchTerm}
                  onChange={(event) => setTextSearchTerm(event.target.value)}
                  placeholder={'e.g. "rain-soaked neon"'}
                />
              </label>
              <div className="action-row">
                <button
                  type="submit"
                  className="primary"
                  disabled={textSearchLoading}
                >
                  {textSearchLoading ? 'Scanning…' : 'Search text'}
                </button>
                {textSearchError && <p className="error">{textSearchError}</p>}
              </div>
            </form>
            <div className="tool-results-panel">
              <div className="tool-results-header">
                <div>
                  <p className="eyebrow small">Snippets</p>
                  <h3>Text search</h3>
                </div>
              </div>
              {textMatches.length > 0 ? (
                <ul className="text-match-list scrollable">
                  {textMatches.map((match) => (
                    <li key={`${match.tconst}-${match.snippet.slice(0, 12)}`}>
                      <strong>{match.primary_title ?? match.tconst}</strong>
                      {match.start_year && (
                        <span className="subdued small"> · {match.start_year}</span>
                      )}
                      <p>{match.snippet}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="tool-placeholder">
                  <p>Search Grok analyses or cleaned plots for an exact snippet.</p>
                  <p className="subdued small">We’ll highlight every match we find.</p>
                </div>
              )}
            </div>
          </div>
        );
    }
  };

  const panelMeta = TOOL_META[activeTool];
  const panelHint = panelMeta.panelHint ?? panelMeta.hint;

  return (
    <div className="page">
      <header className="header">
        <div>
          <p className="eyebrow">movie recommender</p>
          <h1>Explorer Console</h1>
          <p className="subdued">
            Blend embeddings, run text scrapes, and inspect stored analyses without
            touching the CLI.
          </p>
        </div>
        <div className="selector-row">
          <label className="selector">
            <span>Profile</span>
            <select
              value={selectedProfile}
              onChange={(event) => setSelectedProfile(event.target.value)}
            >
              {profiles.map((profile) => (
                <option key={profile} value={profile}>
                  {profile}
                </option>
              ))}
            </select>
          </label>
          <label className="selector">
            <span>Index</span>
            <select
              value={selectedIndex}
              onChange={(event) => setSelectedIndex(event.target.value)}
            >
              {indexes.map((index) => (
                <option key={index} value={index}>
                  {index}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <div className="tool-tabs">
        {(Object.keys(TOOL_META) as ToolKey[]).map((tool) => (
          <button
            key={tool}
            type="button"
            className={`tool-tab ${activeTool === tool ? 'active' : ''}`}
            onClick={() => setActiveTool(tool)}
          >
            <span>{TOOL_META[tool].label}</span>
            <small>{TOOL_META[tool].hint}</small>
          </button>
        ))}
      </div>

      <section className="panel focus-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">{panelMeta.eyebrow}</p>
            <h2>{panelMeta.label}</h2>
          </div>
          {panelHint && <p className="subdued small panel-hint">{panelHint}</p>}
        </div>
        {renderActivePanel()}
      </section>
    </div>
  );
}

export default App;

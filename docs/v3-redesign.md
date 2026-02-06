# V3 Redesign: Homepage, NMF Migration, Term Trends, Data Persistence

**Date:** February 6, 2026

---

## What Changed

### Homepage Redesign

The homepage was rebuilt from scratch to focus on communicating value rather than technique.

**Before:** Title was "Claude.md Topic Analyzer" with subtitle "Discovering patterns in Claude documentation across GitHub repositories using advanced topic modeling." Five topic cards showed flat word lists with no visual weight or proportionality. A "Loading visualization..." section at the bottom often failed to render. The privacy notice was prominently placed on the homepage.

**After:** Title is "Claude Context Analysis." The subtitle explains what the tool does in plain language. A CLAUDE.md definition is included for visitors who don't know what these files are. Below the hero, a metrics bar shows repositories analyzed, files processed, topics discovered, and last analysis date. A "Key Findings" callout box programmatically generates insights (most prevalent topic, cross-cutting terms, unique term count, analysis run count). Topic cards now use horizontal CSS bar charts where bar width is proportional to word weight from the model. Each topic has a colored left border, icon, one-line description, and an expandable "What does this topic mean?" section with contextual interpretation. The privacy notice moved to the footer.

### NMF Replaces LDA

The topic modeling algorithm was changed from Latent Dirichlet Allocation (LDA) to Non-negative Matrix Factorization (NMF).

**Why LDA was brittle:**

- Non-deterministic topics. LDA uses random initialization and iterative optimization. Even with a fixed `random_state`, the topics it finds depend heavily on which 500 files GitHub returns that day. The same underlying data can produce different topic groupings across runs.
- Bag of words. LDA treats documents as unordered word counts. It doesn't understand that "npm run test" is a command or that "React component" is a phrase. It just sees individual tokens that happen to co-occur.
- Fixed topic count. The code hardcodes `num_topics=5`. If the real data has 3 natural clusters or 8, LDA will still force it into 5, either merging distinct themes or splitting coherent ones.
- Short documents. Many CLAUDE.md files are short. LDA works best with longer documents where word co-occurrence patterns are more reliable.

**Why NMF is better for this use case:**

- More coherent topics. NMF tends to produce topics where the top words are more obviously related to each other, making the labels more meaningful.
- More stable across runs. Given similar input data, NMF produces more consistent topic compositions than LDA, which helps with topic evolution tracking.
- Better with TF-IDF. NMF pairs naturally with TF-IDF weighting (via `TfidfVectorizer`), which downweights common terms and emphasizes distinctive ones. LDA used raw counts via `CountVectorizer`.
- Same memory footprint. NMF is available in scikit-learn with the same API surface. No new dependencies, no additional memory overhead. The swap was a near drop-in replacement.

**What changed in code:**

- `CountVectorizer` replaced with `TfidfVectorizer`
- `LatentDirichletAllocation` replaced with `NMF`
- `init='nndsvda'` used for deterministic initialization
- `max_iter` increased from 10 to 200 (NMF converges faster per iteration, total time is comparable)
- The rest of the pipeline (preprocessing, topic extraction, visualization) is unchanged because both models expose the same `.components_` matrix

**Alternatives considered:**

- BERTopic: Uses transformer embeddings for semantic understanding. Would produce much better topics but requires 80-400MB for the sentence-transformers model, which won't fit on Render's free tier (512MB RAM).
- Top2Vec: Similar to BERTopic, same memory concern.
- NMF was the pragmatic choice: biggest quality improvement with zero infrastructure change.

### Term Trends Page

A new `/trends` page was added showing how individual terms rise and fall in popularity over time.

**Why term-level tracking instead of topic-level:** LDA/NMF topic labels shift between runs because the algorithms don't guarantee consistent topic ordering or composition. The label generation function (`generate_topic_label`) is a brittle heuristic that pattern-matches the top 5 words against hardcoded keyword lists. If "react" happens to be the 6th word instead of top 5, a clearly frontend topic won't get labeled "Frontend Development." Individual word weights are the actual stable data; topic labels are cosmetic.

The trends page tracks the 15 most frequent terms across all historical runs. For each daily run, it sums each term's weight across all topics in that run and plots the result.

Two charts are provided:
- **Absolute weight chart:** shows raw total weight per term per day. Terms with inherently higher weights (like "npm") dominate visually.
- **Normalized chart:** scales each term to 0-100% based on its own min/max, making it easier to compare relative trends between terms with very different basitudes.

Colored toggle buttons let users show/hide individual terms. The top 6 are active by default.

### Data Persistence

**Problem:** Analysis history was stored in JSON files on Render's ephemeral filesystem. Every redeploy wiped all accumulated data, losing the topic evolution history.

**Solution:** Downloaded 30 days of analysis history (Jan 8 - Feb 6, 2026) from the live Render instance via the `/api/topic-evolution` endpoint, reconstructed the `analysis_history.json` and `last_analysis.json` files, and committed them to a `data/` directory in the repository.

The app now reads/writes from `data/` on both platforms:
- On Render: `/opt/render/project/src/data/` (inside the cloned repo)
- Locally: `data/` (relative path)

New analysis runs append to `analysis_history.json` as before (capped at 30 entries). On the next redeploy, the committed data is the starting point and new runs build on top of it.

### Repo Cleanup

- Removed `venv/` from git tracking (was a full Python 3.8 virtualenv committed to the repo, ~560K lines deleted)
- Removed `analysis.db` from git tracking
- Removed root-level `style.css` and `main.js` (empty files that belonged in `/static`)
- Added `venv/`, `analysis.db`, `logs/`, `cache/` to `.gitignore`
- Added note in `CLAUDE.md` that `app.py` is the canonical entry point (ignoring `app_mock.py`, `app_simple.py`, `app-minimal.py`)

### Other Changes

- New favicon: trend-line chart icon (blue/purple polylines on dark background) replacing the robot head
- `/visualization` route now redirects to homepage instead of showing a 404 when the generated HTML file doesn't exist
- Navigation cards updated with "Term Trends" link and preview text

---

## Architecture Overview

### Data Collection

The app uses the GitHub Code Search API to find files named `claude.md` across public repos. It pages through results (100 per request), downloading each file's content via the download URL or Contents API fallback. It collects up to 500 files per run. Rate limiting is handled with sleeps and retry logic. Requires a `GITHUB_PAT` environment variable.

### Analysis Pipeline

Once files are collected, they go through:

1. **Preprocessing** (`preprocess_text`): lowercasing, regex cleanup, NLTK tokenization, stopword removal, lemmatization
2. **Vectorization**: `TfidfVectorizer` converts the cleaned text into a weighted document-term matrix
3. **NMF**: `NMF` from scikit-learn discovers 5 topics, each a distribution over words
4. **Label generation** (`generate_topic_label`): heuristic that looks at the top words and assigns a human-readable name like "Frontend Development"

The output is two things:
- A standalone HTML visualization file written to `data/lda_visualization.html` (served by `/visualization`)
- A topics data structure (labels, top words, weights, strengths) saved to JSON

### Label Generation

Labels are assigned to entire topics, not individual words. Each NMF topic is a distribution over all words in the vocabulary. The function checks the top 5 words against hardcoded keyword lists in priority order:

```python
if any(word in top_5_words for word in ['npm', 'typescript', 'react', 'pnpm']):
    return "Frontend Development"
elif any(word in top_5_words for word in ['code', 'function', 'class']):
    return "Code Guidelines"
# ... more elif chains ...
else:
    return f"{top_words[0].title()} & {top_words[1].title()}"
```

If none of the patterns match, it falls back to joining the first two words (e.g. "Event & State"). This is the weakest part of the pipeline, which is why the term trends page tracks individual words instead of relying on these labels.

### Storage

No database. Everything is JSON files in `data/`:

- **`last_analysis.json`**: the most recent run's results (timestamp, success flag, files collected, full topics data with words and weights)
- **`analysis_history.json`**: array of all runs (up to 30), used for topic evolution and term trends

These files are committed to the repo so they survive Render redeploys. New analysis runs append to the history file.

### Scheduling

`start_analysis_scheduler()` launches a daemon thread on app startup that checks every hour if it's time to run. `should_run_analysis()` checks if it's 3 AM GMT and no analysis has run since yesterday. When triggered, `run_analysis_now()` spawns another thread that runs the full collection + analysis pipeline.

### Flask Routes

**Pages:**
- `/` - homepage; reads from `last_analysis.json` to populate metrics, key findings, and topic bar charts via Jinja2 server-side rendering
- `/how-it-works` - static explainer page
- `/topic-evolution` - Chart.js page showing topic stability across runs, fetches data client-side
- `/trends` - Chart.js page showing term popularity over time with toggleable terms
- `/visualization` - serves the generated standalone HTML file, redirects to `/` if it doesn't exist

**APIs (called client-side by the Chart.js pages):**
- `/api/term-trends` - processes history, returns per-term weight time series for the top 15 terms
- `/api/topic-evolution` - runs topic similarity matching across historical runs using cosine similarity
- `/api/topics-3d` - returns current topic data (with hardcoded fallback if no real data)
- `/api/database-stats` - returns run counts from history
- `/analyze` - GET returns status, POST triggers manual analysis

### Deployment

On Render free tier: `gunicorn app:app` starts the app, the scheduler thread kicks off, and it runs the daily analysis at 3 AM GMT. The `data/` directory lives at `/opt/render/project/src/data` (inside the cloned repo). Between deploys, new analysis runs append to the JSON files. On redeploy, the committed data is the starting point.

# Data guide

Where the corpus comes from, how it was sampled, and what it therefore cannot be used to claim.

## Source

Paper metadata is fetched from the [public arXiv API](https://info.arxiv.org/help/api/index.html) — an Atom feed served at `https://export.arxiv.org/api/query`. No account, no bulk download, no Kaggle snapshot: the API is live, so rebuilding the corpus next month picks up next month's papers.

Only metadata is used — title, abstract, submission date, categories. Full texts are neither fetched nor needed.

arXiv metadata is available for reuse; the project follows the API's usage rules, which is what the three-second pause between requests and the explicit `Retry-After` handling are for.

## What was collected

| | |
|---|---|
| Categories | `cs.CL`, `cs.CV`, `cs.CR`, `cs.RO`, `cs.DB` |
| Period | January 2018 – June 2026 (102 months) |
| Quota | 40 papers per category per month |
| Cross-listed papers | excluded; only the primary category counts |
| Documents | 20,169 (of a 20,400 quota) |
| Median per month | 200, range 184–200 |
| Mean length | 1,292 characters |

The five categories were chosen to be far apart — language, vision, security, robotics, databases — so that agreement between discovered topics and real labels means something. Picking `cs.LG` and `stat.ML`, which are cross-posted together almost by default, would have made any model look bad for correctly merging them.

Text is the **title followed by the abstract**: titles of scientific papers are dense and improve coherence. This is controlled by `data.arxiv.include_title`.

## Schema

Every source in this project produces the same five columns, which is what makes the arXiv API and a user CSV interchangeable:

| Column | Meaning |
|---|---|
| `doc_id` | arXiv identifier, version stripped (`2401.01234`) |
| `text` | what the models see: title + abstract |
| `title` | shown in the UI |
| `label` | primary arXiv category, used as the reference partition |
| `date` | submission date, drives the timeline |

## Filtering

Applied once, in `finalize_corpus`:

- whitespace normalized, empty documents dropped;
- abstracts shorter than 250 characters dropped as uninformative;
- duplicates removed twice over — by identifier, and by identical text, since papers withdrawn and resubmitted under a new number carry byte-identical abstracts. On this corpus the check found none, which is itself worth knowing: it means the identifier deduplication in the fetch layer already did the job.

## Sampling, and what it biases

**The corpus is a quota sample, not a census.** Each category contributes exactly 40 papers a month, so the fields are held at equal size regardless of how much each actually publishes. `cs.CV` puts out an order of magnitude more papers than `cs.DB`, and none of that is visible here.

This is deliberate. It is what makes the topic-shift result clean: when the LLM topic grows from 4% to 25% of the corpus, that cannot be an artifact of NLP publishing more, because the number of NLP papers is fixed. It also means the corpus **cannot** answer how fast a field grew.

**Within a month, papers are the earliest submissions.** Results are sorted by submission date for reproducibility — without an explicit order arXiv may paginate differently between runs — and the quota is filled from the top. How wide that window is depends on how busy the category is:

| Category | Mean day of month | Median | Max |
|---|---|---|---|
| `cs.CV` | 1.3 | 1 | 4 |
| `cs.CL` | 2.1 | 1 | 13 |
| `cs.RO` | 3.2 | 2 | 22 |
| `cs.CR` | 3.3 | 3 | 15 |
| `cs.DB` | 12.4 | 11 | 31 |

So for vision the sample is effectively the first day or two of each month, while for databases it spans half of it. A conference deadline falling mid-month would be missed for the busy categories.

Two things keep this from undermining the results. All three models see exactly the same corpus, so the comparison between them is unaffected. And the headline trend was verified independently: the share of abstracts that mention LLMs at all rises 0% → 1.2% → 8.9% → 22.4% → 31% across 2018–2026, matching what the models found.

A proper fix would be a random offset within each month, with the seed recorded in the cache fingerprint so the draw stays reproducible. It costs another full fetch and is not done yet.

**The `cs.DB` quota is not always met.** In the early years the category does not publish 40 papers in some months, which is where most of the 231 missing documents come from.

## Rebuilding the corpus

```bash
python scripts/fetch_data.py --limit-per-slice 3   # 510 requests, small payloads
python scripts/fetch_data.py                       # the real thing, about an hour
```

The full fetch issues one request per (category, month) plus retries, with a three-second pause between them. It is designed to be interrupted: each completed slice is appended to a JSONL checkpoint under `data/raw/slices/`, keyed by a hash of the query, and a rerun resumes from where it stopped.

The assembled corpus is cached as parquet under `data/cache/`, named with a hash of everything that shaped it — categories, dates, quotas, filters. Change any of them and you get a new file rather than a stale one. A JSON sidecar records the parameters, the timestamp and a summary.

Neither directory is committed: the corpus is 15 MB and reproducible from the API.

## The demo corpus

`data/demo/arxiv-demo.csv.gz` (1.1 MB, 2,487 documents) ships with the repository so the hosted app starts without training. It is a stratified sample of the full corpus, drawn proportionally within each (category, year) cell, so the category balance and the full date range survive the shrinking.

It is deliberately a **CSV**, read by the same `CsvSource` any user's data would go through — the public demo therefore doubles as proof that the pipeline is not arXiv-specific.

Because it holds an eighth of the documents, its metrics are lower than the ones quoted in the README: NMF purity 0.735 against 0.816, LDA 0.540 against 0.706. Fewer documents, less stable topics; this is the expected direction.

Rebuild it with `python scripts/make_demo.py`.

## Using your own data

Point the config at a CSV and name the columns:

```yaml
data:
  source: csv
  min_abstract_chars: 100      # lower this for short documents
  csv:
    path: data/raw/reviews.csv
    text_column: body
    label_column: product      # optional: enables NMI, ARI and purity
    date_column: published     # optional: enables the timeline
    id_column: review_id       # optional: row numbers are used otherwise
```

Two practical limits. Topic models need **hundreds of documents at minimum, thousands to be stable** — topics are inferred from how words co-occur across documents, so a small file produces noise rather than structure; the app warns below 200. And preprocessing is **English-only**: an English stopword list and an English lemmatizer.

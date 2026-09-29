**[Russian Version / На русском](README.ru.md)**

# TopicLens

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CI](https://github.com/IlyaShaposhnikov/topic-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/IlyaShaposhnikov/topic-lens/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**LDA, NMF and LSA fitted on the same corpus and compared on coherence, topic diversity and agreement with real category labels — not on how plausible their top words look.**

Most topic-modeling projects fit one model and print ten word lists. This one fits three, measures them against each other, matches their topics pairwise, and tracks how the topics move over eight years of arXiv abstracts.

**[Live demo](https://topic-lens.streamlit.app/)** · built from 20,169 abstracts across five arXiv categories, January 2018 – June 2026.

## What it answers

| Question | How |
|---|---|
| Which model produces better topics? | NPMI and UMass coherence, topic diversity, pairwise overlap |
| Do the topics mean anything real? | Agreement with the paper's own arXiv category (NMI, ARI, purity) |
| Did the three models find the same structure? | Cosine similarity of topic vectors, matched with the Hungarian algorithm |
| How many topics should there be? | Coherence and diversity swept across k, with seed noise measured |
| What changed in eight years? | Topic shares per quarter, over the whole corpus |

## Results

Full corpus, 8 topics per model, identical preprocessing, seed fixed.

| | LDA | NMF | LSA |
|---|---|---|---|
| **NPMI coherence** ↑ | 0.112 | **0.196** | 0.118 |
| **UMass coherence** ↑ | −1.891 | **−1.656** | −2.004 |
| **Topic diversity** ↑ | 0.813 | **0.850** | 0.638 |
| **Pairwise overlap** ↓ | 0.046 | **0.029** | 0.084 |
| **NMI vs categories** ↑ | 0.395 | **0.548** | 0.083 |
| **ARI vs categories** ↑ | 0.381 | **0.524** | 0.004 |
| **Purity** ↑ | 0.706 | **0.816** | 0.253 |
| **Fit time** | 286 s | 46 s | **1.3 s** |

**NMF wins every quality metric.** Its topics are the most coherent, the least overlapping, and by far the closest to the real categories: in 82% of cases a paper's dominant topic matches the majority category of that topic.

**LSA is not a topic model, and the numbers say so plainly.** Purity of 0.253 sits barely above the 0.20 that random assignment gives on five balanced categories, and ARI of 0.004 means no agreement at all once chance is discounted. Its first component is not a topic but "the average scientific abstract" — the direction of greatest variance, which is what SVD is designed to find. It is kept in the comparison precisely because this failure is informative, and it earns its place in the app as a **similarity search** engine, which is what the method is genuinely good at.

**Two models out of three agree with each other.** Mean similarity of optimally matched topics: LDA↔NMF **0.680**, LSA↔NMF 0.388, LDA↔LSA 0.314. LDA and NMF independently found the same computer-vision topic (cosine 0.91, eight shared words out of ten) and the same robotics topic (0.89), despite being fitted on different matrices with different objectives.

## The topics move

![Topic shares over time](docs/images/timeline-nmf.png)

The large-language-model topic grows from 4% of the corpus to 25%, with the break in 2023. Security follows it upward — work on attacks against LLMs.

**This is a shift in what papers are about, not growth in publication volume.** The corpus holds exactly 40 papers per category per month by design, so the size of each field is held constant. The same 40 monthly `cs.CL` papers were about machine translation and word embeddings in 2018 and about large language models in 2025.

Verified independently of the models, by counting abstracts that mention LLMs at all:

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| 0.0% | 0.0% | 0.2% | 0.3% | 1.2% | 8.9% | 22.4% | 28.1% | 31.0% |

The curve matches what the model found without being told anything about LLMs.

## What the experiments showed

Every parameter below was chosen by measurement, and the measurements are as interesting as the choices.

**The number of topics is mostly noise for LDA.** Sweeping k from 5 to 20 moved LDA's coherence by 0.022 — while changing only the random seed moved it by 0.023. Picking k from that curve would have been reading tea leaves. NMF is the opposite: with `nndsvda` initialization (itself an SVD of the matrix) it is fully deterministic across seeds, so its differences are real, and it peaks at k = 5–8. Eight topics are used for all three models, so that the comparison stays like-for-like.

**A bigger vocabulary made things worse.** The full 1–2-gram vocabulary holds 55,986 features, of which 44,860 are bigrams. Capping it at 20,000 — which keeps the unigrams and the most frequent bigrams — improved NMF's coherence from 0.193 to 0.217, diversity from 0.850 to 0.887 and purity from 0.807 to 0.831. The long tail of rare bigrams is noise, and the cap is a deliberate filter rather than an accident.

**Domain stopwords were chosen by looking at the data.** Scientific abstracts share a fixed rhetorical vocabulary — *propose*, *demonstrate*, *state-of-the-art*, *outperform* — which would otherwise top every topic. They were selected from a document-frequency ranking, while substantive terms were deliberately kept even at high frequency: `data` appears in 41% of abstracts and is exactly what the database topic is made of.

**LDA never reaches its convergence threshold**, stopping at the iteration limit. Note that scikit-learn evaluates convergence only when `evaluate_every` is positive — left at its default, `n_iter_` always equals `max_iter` and tells you nothing.

## Two bugs worth mentioning

**All 900 papers came from January.** The first corpus was sliced by year, and combined with sorting by submission date that meant the first days of each January — which on arXiv is when everything prepared for the ICLR and AAAI deadlines lands. The topic timeline would have been four points per year with one of them filled. Fixed by slicing per month; the sampling still favours the first days of each month, which is documented in [DATA_GUIDE.md](docs/DATA_GUIDE.md).

**Half the LSA topics showed their least characteristic words.** The sign of a singular vector is arbitrary — a component and its negation span the same subspace — so roughly half of them had their dominant mass on the negative side. Components are now oriented before their top words are read.

## Architecture

```
topic-lens/
├── configs/config.yaml          # every parameter of every step, validated on load
├── topiclens/
│   ├── config.py                # pydantic models; unknown keys are errors
│   ├── constants.py             # paths, model keys, category names
│   ├── data/
│   │   ├── arxiv.py             # API client: paging, retries, rate limits
│   │   ├── base.py              # corpus schema and the source protocol
│   │   ├── corpus.py            # assembly, parquet cache, resumable checkpoints
│   │   ├── csv_source.py        # bring your own data
│   │   └── sample.py            # stratified sampling for the demo corpus
│   ├── preprocessing.py         # LaTeX-aware cleaning, lemmatization, vectorizers
│   ├── models/
│   │   ├── base.py              # the shared TopicModel contract
│   │   ├── lda.py  nmf.py  lsa.py
│   │   └── factory.py           # build models from config
│   ├── evaluation/
│   │   ├── coherence.py         # NPMI and UMass, implemented directly
│   │   ├── diversity.py         # distinctness of the topic set
│   │   ├── alignment.py         # agreement with the corpus labels
│   │   ├── matching.py          # Hungarian matching across models
│   │   └── selection.py         # the topic-count sweep
│   ├── pipeline.py              # preprocess → vectorize → fit → evaluate
│   ├── artifacts.py             # the bundle that keeps training and inference identical
│   ├── reporting.py             # metrics, topics, timeline and matches as files
│   ├── viz/charts.py            # every Plotly figure used anywhere
│   └── ui/                      # app logic (data.py) separated from rendering (views.py)
├── scripts/
│   ├── fetch_data.py            # build the corpus
│   ├── train.py                 # train, evaluate, report
│   ├── make_demo.py             # the small corpus and bundle that ship with the repo
│   └── setup_nltk.py            # one-time NLTK data
├── streamlit_app.py             # the app
└── tests/                       # 240+ tests, no network required
```

Three decisions shape the rest.

**One contract for three algorithms.** `TopicModel` exposes `fit`, `transform`, `top_words` and `document_topics`; everything downstream works against the base class. Where the algorithms genuinely differ, the difference is explicit rather than hidden: `document_topics` clips the negative part of the SVD loadings so that "share of a topic in a document" means the same thing for all three, and each model reports its own native diagnostics separately, because perplexity and reconstruction error are not comparable.

**The bundle keeps training and inference identical.** Preprocessing here is a separate step, not something hidden inside the vectorizer — which is what keeps a fitted vectorizer from depending on this package at unpickling time. The cost is that a new text could be tokenized differently at inference. `ModelBundle.transform_text` removes that risk by routing every new document through the exact objects the models were fitted with.

**Sources are interchangeable.** Anything that produces `doc_id, text, title, label, date` is a corpus. The arXiv API and a user-supplied CSV go through the same filtering, the same cache and the same pipeline — which is why the hosted demo reads its corpus from a CSV.

## Engineering notes

**The corpus fetch survives being interrupted.** 510 requests over roughly an hour will fail somewhere. Each completed slice is appended to a JSONL checkpoint keyed by a hash of the query, so a rerun resumes instead of starting over — which it did, after a rate limit killed run #1 at slice 462 of 510.

**HTTP 429 is handled separately from other errors.** Rate limits need minutes, not the ordinary exponential backoff, and `Retry-After` is honoured when present. arXiv also returns empty feeds while reporting matches, which is a reason to retry rather than to stop.

**The cache invalidates itself.** Its filename contains a hash of everything that shapes the corpus — categories, date range, quotas, filters. Change a parameter and you get a new file, not a silently stale one; a JSON sidecar records what produced it and when.

**Coherence is implemented directly** rather than through gensim: twenty lines of co-occurrence arithmetic, no heavy dependency, and a definition anyone can audit.

## Quick start

```bash
git clone https://github.com/IlyaShaposhnikov/topic-lens.git
cd topic-lens

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/setup_nltk.py
```

Run the app straight away — it falls back to the demo bundle committed in the repository:

```bash
streamlit run streamlit_app.py
```

Build the full corpus and train on it (the fetch takes about an hour of polite, rate-limited requests):

```bash
python scripts/fetch_data.py --limit-per-slice 3    # a quick trial first
python scripts/fetch_data.py                        # ~20k abstracts
python scripts/train.py                             # ~6 minutes, mostly LDA
python scripts/train.py --sample 2000               # or iterate on a subset
python scripts/train.py --sweep                     # also sweep the topic count
```

Use your own data by pointing the config at a CSV:

```yaml
data:
  source: csv
  csv:
    path: data/raw/my-documents.csv
    text_column: body
    label_column: category     # optional, enables the agreement metrics
    date_column: published     # optional, enables the timeline
```

Any value can also be overridden from the environment, which is what the CI and the hosted app use:

```bash
TOPICLENS_MODELS__N_TOPICS=12 python scripts/train.py
```

## Testing

240+ tests, none of which touch the network: the arXiv client is exercised against a fake HTTP session that serves canned Atom feeds, including truncated XML, HTML error pages, empty feeds and rate-limit responses.

```bash
pytest                      # everything
pytest -m "not slow"        # skips the tests that fit real models
```

CI runs the linter separately from the tests, the tests on Python 3.10, 3.11 and 3.12, and a dependency audit on `main` and weekly — deliberately not on pull requests, so a fresh advisory in a transitive dependency cannot block an unrelated change.

## Limitations

- **The corpus is a quota sample, not a census.** 40 papers per category per month, taken from the earliest submissions of that month. Fields are held at equal size, which is what makes the topic-shift result clean, but it means the data cannot answer how fast a field grew.
- **Eight topics over five categories** is a choice of granularity, not a truth. Categories and topics do not correspond one-to-one in either direction: `cs.CL` splits into several topics, while adversarial attacks run across three categories.
- **Topic modeling needs a corpus.** Topics come from how words co-occur across many documents, so a single text cannot produce them — it can only be scored against models already trained on a corpus. The app warns when an uploaded file holds fewer than 200 documents.
- **English only.** The preprocessing assumes it: an English stopword list and an English lemmatizer.

## Author

Ilya Shaposhnikov | [E-mail](mailto:ilia.a.shaposhnikov@gmail.com) | [LinkedIn](https://linkedin.com/in/iliashaposhnikov)

Data: [arXiv API](https://info.arxiv.org/help/api/index.html). Thank you to arXiv for use of its open access interoperability.

**[Russian Version / На русском](README.ru.md)**

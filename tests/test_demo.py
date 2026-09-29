"""Guards on the demo artifacts that ship with the repository.

These files are committed binaries, so they drift silently when the code around
them changes. The tests skip when the demo has not been built, and fail loudly
when what is committed can no longer be loaded — which is exactly the failure a
visitor to the hosted app would hit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from topiclens.artifacts import ARTIFACT_VERSION, BUNDLE_FILENAME, ModelBundle
from topiclens.constants import MODEL_KEYS, PROJECT_ROOT
from topiclens.ui.data import MIN_CORPUS_DOCUMENTS, load_reports

DEMO_ARTIFACTS = PROJECT_ROOT / "artifacts" / "demo"
DEMO_CORPUS = PROJECT_ROOT / "data" / "demo" / "arxiv-demo.csv.gz"

needs_demo = pytest.mark.skipif(
    not (DEMO_ARTIFACTS / BUNDLE_FILENAME).is_file(),
    reason="demo artifacts are not built; run scripts/make_demo.py",
)


@needs_demo
def test_committed_bundle_still_loads():
    bundle = ModelBundle.load(DEMO_ARTIFACTS)
    assert bundle.version == ARTIFACT_VERSION
    assert set(bundle.models) == set(MODEL_KEYS)


@needs_demo
def test_committed_bundle_can_score_a_text():
    shares = ModelBundle.load(DEMO_ARTIFACTS).transform_text(
        "We study reinforcement learning for robot manipulation from demonstrations."
    )
    assert set(shares) == set(MODEL_KEYS)
    assert all(abs(values.sum() - 1.0) < 1e-6 for values in shares.values())


@needs_demo
def test_demo_reports_are_present():
    reports = load_reports(DEMO_ARTIFACTS)
    assert set(reports.metrics) == set(MODEL_KEYS)
    assert reports.has_timeline


@pytest.mark.skipif(not DEMO_CORPUS.is_file(), reason="demo corpus is not built")
def test_demo_corpus_is_large_enough_and_balanced():
    frame = pd.read_csv(DEMO_CORPUS, parse_dates=["date"])
    assert len(frame) >= MIN_CORPUS_DOCUMENTS

    shares = frame["label"].value_counts(normalize=True)
    assert shares.max() - shares.min() < 0.05
    assert frame["date"].dt.year.nunique() >= 5

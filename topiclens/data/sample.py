"""Drawing a smaller corpus that still looks like the big one.

A demo corpus has to fit in a repository and in a free hosting tier, but it must
not distort what the app shows. Taking the first N documents would flatten the
category balance and cut the time axis short — and the time axis is the point of
half the charts. Sampling proportionally within each (label, period) cell keeps
both, at a fraction of the size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)


def stratified_sample(
    frame: pd.DataFrame,
    size: int,
    *,
    seed: int = 0,
    period_freq: str = "Y",
) -> pd.DataFrame:
    """Sample ``size`` documents, preserving the label and time composition.

    Each (label, period) cell contributes in proportion to its share of the
    corpus, with at least one document per non-empty cell so that no category
    and no period disappears entirely.

    Args:
        frame: the full corpus.
        size: how many documents to keep; a larger value returns everything.
        seed: fixes the draw, so the demo corpus is reproducible.
        period_freq: pandas period alias used to stratify over time.

    Raises:
        ValueError: if ``size`` is not positive.
    """
    if size <= 0:
        raise ValueError(f"sample size must be positive, got {size}")
    if size >= len(frame):
        return frame.reset_index(drop=True)

    keys = []
    if frame["label"].notna().any():
        keys.append(frame["label"].fillna("unlabelled"))
    if frame["date"].notna().any():
        keys.append(frame["date"].dt.to_period(period_freq).astype(str))
    if not keys:
        return frame.sample(size, random_state=seed).reset_index(drop=True)

    cells = pd.Series(list(zip(*[key.to_numpy() for key in keys], strict=True)), index=frame.index)
    fraction = size / len(frame)

    generator = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for _, indices in frame.groupby(cells).groups.items():
        quota = max(1, int(round(len(indices) * fraction)))
        quota = min(quota, len(indices))
        selected.append(generator.choice(np.asarray(indices), size=quota, replace=False))

    picked = np.concatenate(selected)
    if len(picked) > size:
        picked = generator.choice(picked, size=size, replace=False)

    result = frame.loc[np.sort(picked)].reset_index(drop=True)
    logger.info(
        "Sampled %d of %d documents across %d cells", len(result), len(frame), len(selected)
    )
    return result

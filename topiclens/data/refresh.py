"""Working out what a scheduled refresh should fetch.

The demo corpus grows month by month rather than being rebuilt: a full rebuild
means an hour of requests from a shared CI address, while an increment is five.
The arithmetic of "which months are missing and how many papers to take" is kept
here, apart from the script that performs the fetch, so it can be tested without
touching the network.
"""

from __future__ import annotations

import pandas as pd

from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)

#: Never fetch fewer than this per (category, month); below it a month barely
#: registers on the timeline.
MIN_QUOTA = 3


def last_complete_month(today: pd.Timestamp | None = None) -> str:
    """The most recent month that has finished, as ``YYYY-MM``.

    The current month is deliberately excluded: a partial month would show up on
    the timeline as a dip that says nothing about the corpus.
    """
    moment = pd.Timestamp.utcnow() if today is None else pd.Timestamp(today)
    return str(moment.tz_localize(None).to_period("M") - 1)


def latest_period(frame: pd.DataFrame) -> str | None:
    """The newest month present in a corpus, or ``None`` if it carries no dates."""
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    return str(dates.max().to_period("M")) if not dates.empty else None


def missing_periods(frame: pd.DataFrame, *, until: str | None = None) -> list[str]:
    """Months between the end of the corpus and the last complete month."""
    boundary = until or last_complete_month()
    newest = latest_period(frame)
    if newest is None or newest >= boundary:
        return []

    periods = pd.period_range(
        start=pd.Period(newest, freq="M") + 1, end=pd.Period(boundary, freq="M"), freq="M"
    )
    return [str(period) for period in periods]


def slice_quota(frame: pd.DataFrame) -> int:
    """How many papers a new (category, month) slice should hold.

    Matched to the density of what is already there. Fetching a full quota into
    a corpus that was sampled down would make recent months several times denser
    than older ones, tilting every share on the timeline.
    """
    dates = pd.to_datetime(frame["date"], errors="coerce")
    cells = frame.assign(period=dates.dt.to_period("M")).groupby(["label", "period"], observed=True)
    counts = cells.size()
    quota = max(MIN_QUOTA, int(round(counts.median()))) if not counts.empty else MIN_QUOTA
    logger.info("New slices will hold %d papers, matching the existing density", quota)
    return quota


def merge_corpora(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Append new documents, dropping anything already present, keeping date order."""
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset="doc_id", keep="first")
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    return combined.sort_values("date", ignore_index=True)

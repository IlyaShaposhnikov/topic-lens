"""Project-wide constants: paths, naming conventions, reference tables.

Anything that several modules need to agree on lives here, so that a rename
happens in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root, derived from this file's location (``topiclens/`` is one
#: level below it). Every relative path from the config is resolved against it,
#: so the CLI behaves the same no matter which directory it is called from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"

#: Prefix for environment overrides, e.g. ``TOPICLENS_MODELS__N_TOPICS=15``.
ENV_PREFIX = "TOPICLENS_"
ENV_NESTING_SEPARATOR = "__"

#: Supported model keys, in the order they are reported and plotted.
MODEL_KEYS = ("lda", "nmf", "lsa")

#: Human-readable labels for the arXiv categories used by default.
ARXIV_CATEGORY_NAMES = {
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision",
    "cs.CR": "Cryptography and Security",
    "cs.RO": "Robotics",
    "cs.DB": "Databases",
    "cs.LG": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "cs.IR": "Information Retrieval",
    "cs.SE": "Software Engineering",
    "stat.ML": "Machine Learning (Statistics)",
}


def resolve_path(path: Path | str) -> Path:
    """Return an absolute path, resolving relative ones against the repo root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

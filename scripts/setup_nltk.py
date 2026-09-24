"""Download the NLTK data required by topic-lens.

Run once after installing the project:
    python scripts/setup_nltk.py
"""

from __future__ import annotations

from topiclens.utils.nltk_data import NLTK_RESOURCES, ensure_nltk_resources


def main() -> int:
    downloaded = ensure_nltk_resources(quiet=False)
    if downloaded:
        print(f"Downloaded: {', '.join(downloaded)}")
    else:
        print(f"Already present: {', '.join(NLTK_RESOURCES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

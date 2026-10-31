"""Install the fixed public NLTK resources used by GPT-SoVITS.

Explicit official URLs also work with local DNS proxies whose synthetic IPs
are rejected by NLTK's generic URL downloader.
"""

from io import BytesIO
import os
from pathlib import Path
from zipfile import ZipFile

import nltk
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "GPT-SoVITS" / ".codex" / "runtime-cache" / "nltk_data"
RESOURCES = (
    ("corpora", "cmudict"),
    ("tokenizers", "punkt"),
    ("tokenizers", "punkt_tab"),
    ("taggers", "averaged_perceptron_tagger"),
    ("taggers", "averaged_perceptron_tagger_eng"),
)
ARCHIVE_USERS = {"cmudict", "averaged_perceptron_tagger"}


def main():
    directory = Path(os.environ.get("NLTK_DATA", str(DEFAULT_DATA)))
    directory.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(directory))
    for category, name in RESOURCES:
        archive_path = directory / category / f"{name}.zip"
        try:
            nltk.data.find(f"{category}/{name}/")
            if name not in ARCHIVE_USERS or archive_path.is_file():
                continue
        except LookupError:
            pass
        url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/{category}/{name}.zip"
        print(f"Downloading NLTK {name}", flush=True)
        response = requests.get(url, timeout=(15, 120))
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            archive.extractall(directory / category)
        # g2p_en and g2pk2 explicitly look up these archive paths on import.
        if name in ARCHIVE_USERS:
            archive_path.write_bytes(response.content)
    assert nltk.corpus.cmudict.words()
    assert nltk.word_tokenize("Hello, world!")
    assert nltk.pos_tag(["Hello"])
    import pyopenjtalk

    assert pyopenjtalk.g2p("test")
    print(f"Language resources ready: {directory}")


if __name__ == "__main__":
    main()

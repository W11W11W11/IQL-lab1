#!/usr/bin/env python3
"""
Tokenize extracted DCEP-2013 plain-text files and produce word-frequency-length tables.

Expected directory structure (per language):
    <input_root>/
        <LANG>/
            OQ/   *.txt   (oral questions)
            QT/   *.txt   (question time)
            REPORT/       <- IGNORED (not present in all languages)

Output (one file per language, saved to --out-dir):
    <iso>.txt  with three tab-separated columns:
        word_form  frequency  length

Usage
-----
    python tokenize_dcep.py --input dcep_downloads/ --out-dir data/
    python tokenize_dcep.py --input dcep_downloads/ --langs EN FR FI HU MT GA BG
    python tokenize_dcep.py --input dcep_downloads/ --no-spacy   # fastest option

Requirements
------------
    pip install spacy tqdm
    python -m spacy download en_core_web_sm   # repeat for each language
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[warn] tqdm not installed — no progress bars. Run: pip install tqdm", file=sys.stderr)

# ---------------------------------------------------------------------------
# spaCy model map.  Only the tokenizer component is loaded (no NLP pipeline).
# Languages not listed here fall back to the fast regex tokenizer.
# DOCUMENT which languages used the fallback in your Methods section.
# ---------------------------------------------------------------------------
SPACY_MODELS: dict[str, str] = {
    "BG": "bg_core_news_sm",
    "DA": "da_core_news_sm",
    "DE": "de_core_news_sm",
    "EL": "el_core_news_sm",
    "EN": "en_core_web_sm",
    "ES": "es_core_news_sm",
    "FI": "fi_core_news_sm",
    "FR": "fr_core_news_sm",
    "IT": "it_core_news_sm",
    "LT": "lt_core_news_sm",
    "NL": "nl_core_news_sm",
    "PL": "pl_core_news_sm",
    "PT": "pt_core_news_sm",
    "RO": "ro_core_news_sm",
    "SL": "sl_core_news_sm",
    "SV": "sv_core_news_sm",
    # No spaCy model available -> regex fallback:
    # CS, ET, GA, HU, LV, MT, SK
}

# ---------------------------------------------------------------------------
# Regex tokenizer — fast, Unicode-aware, handles all scripts.
# Matches any run of Unicode letters, no digits or punctuation.
# ---------------------------------------------------------------------------
_ALPHA_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

def regex_tokenize(text: str) -> list[str]:
    return _ALPHA_RE.findall(text.lower())

# ---------------------------------------------------------------------------
# spaCy tokenizer-only (cached per model; ~20x faster than full pipeline)
# ---------------------------------------------------------------------------
_nlp_cache: dict[str, object] = {}

def spacy_tokenize(text: str, lang: str) -> list[str]:
    try:
        import spacy
    except ImportError:
        return regex_tokenize(text)

    model_name = SPACY_MODELS.get(lang)
    if not model_name:
        return regex_tokenize(text)

    if model_name not in _nlp_cache:
        try:
            nlp = spacy.load(
                model_name,
                disable=["parser", "ner", "lemmatizer", "tagger",
                         "morphologizer", "senter", "sentencizer", "attribute_ruler"],
            )
            _nlp_cache[model_name] = nlp
            print(f"\n  [spaCy] loaded tokenizer '{model_name}' for {lang}", file=sys.stderr)
        except OSError:
            print(
                f"\n  [warn] spaCy model '{model_name}' not installed for {lang}; "
                f"using regex fallback.\n"
                f"         Fix: python -m spacy download {model_name}",
                file=sys.stderr,
            )
            _nlp_cache[model_name] = None

    nlp = _nlp_cache[model_name]
    if nlp is None:
        return regex_tokenize(text)

    return [token.lower_ for token in nlp(text) if token.is_alpha]

# ---------------------------------------------------------------------------
# File discovery — OQ and QT only, REPORT ignored
# ---------------------------------------------------------------------------
INCLUDE_SUBDIRS = {"OQ", "QT"}

def find_lang_files(lang_root: Path) -> list[Path]:
    files = []
    for subdir in INCLUDE_SUBDIRS:
        d = lang_root / subdir
        if d.is_dir():
            files.extend(sorted(d.glob("*.txt")))
    return files

def find_languages(input_root: Path) -> dict[str, list[Path]]:
    lang_map: dict[str, list[Path]] = {}
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        code = child.name.upper()
        if not re.match(r"^[A-Z]{2}$", code):
            continue
        files = find_lang_files(child)
        if files:
            lang_map[code] = files
        else:
            print(f"  [skip] {code}: no OQ/QT .txt files found", file=sys.stderr)
    return lang_map

# ---------------------------------------------------------------------------
# Process one language
# ---------------------------------------------------------------------------
def process_language(
    lang: str,
    files: list[Path],
    min_freq: int,
    use_spacy: bool,
) -> list[tuple[str, int, int]]:
    counts: Counter = Counter()

    file_iter = (
        tqdm(files, desc=f"  {lang}", unit="file", leave=False)
        if HAS_TQDM else files
    )

    for fpath in file_iter:
        text = fpath.read_text(encoding="utf-8", errors="replace")
        tokens = spacy_tokenize(text, lang) if use_spacy else regex_tokenize(text)
        counts.update(tokens)

    rows = [
        (form, freq, len(form))
        for form, freq in counts.items()
        if freq >= min_freq
    ]
    rows.sort(key=lambda x: -x[1])
    return rows

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
def write_table(rows: list[tuple[str, int, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("word_form\tfrequency\tlength\n")
        for word, freq, length in rows:
            f.write(f"{word}\t{freq}\t{length}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Tokenize DCEP plain-text files -> word/frequency/length tables."
    )
    p.add_argument("--input", required=True,
                   help="Root dir with per-language folders (e.g. dcep_downloads/).")
    p.add_argument("--out-dir", default="data",
                   help="Output directory for <iso>.txt files (default: data/).")
    p.add_argument("--langs", nargs="*", default=None,
                   help="Language codes to process. Default: all found.")
    p.add_argument("--min-freq", type=int, default=1,
                   help="Min token frequency to include (default: 1).")
    p.add_argument("--no-spacy", action="store_true",
                   help="Use fast regex tokenizer for ALL languages (no spaCy). "
                        "Fastest option — document in Methods if used.")
    args = p.parse_args()

    input_root = Path(args.input)
    out_dir = Path(args.out_dir)

    lang_map = find_languages(input_root)
    if not lang_map:
        print(f"[error] No language folders found under {input_root}.", file=sys.stderr)
        return 1

    if args.langs:
        requested = {l.upper() for l in args.langs}
        missing = requested - set(lang_map)
        if missing:
            print(f"[warn] Not found: {sorted(missing)}", file=sys.stderr)
        lang_map = {l: f for l, f in lang_map.items() if l in requested}

    use_spacy = not args.no_spacy
    print(f"Languages : {sorted(lang_map)}")
    print(f"Tokenizer : {'spaCy tokenizer-only + regex fallback' if use_spacy else 'regex (fast, all langs)'}")
    print(f"Min freq  : {args.min_freq}")
    print(f"Output    : {out_dir}\n")

    lang_iter = (
        tqdm(sorted(lang_map.items()), desc="Overall", unit="lang")
        if HAS_TQDM else sorted(lang_map.items())
    )

    for lang, files in lang_iter:
        mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"\n==> {lang}  ({len(files)} files, {mb:.1f} MB)")

        rows = process_language(lang, files, args.min_freq, use_spacy)

        out_path = out_dir / f"{lang.lower()}.txt"
        write_table(rows, out_path)

        n_tokens = sum(f for _, f, _ in rows)
        print(f"    types: {len(rows):,}  |  tokens: {n_tokens:,}  ->  {out_path}")

    tokenizer_desc = (
        "spaCy (tokenizer only, no pipeline) with regex fallback for CS/ET/GA/HU/LV/MT/SK"
        if use_spacy
        else "regex [^\\W\\d_]+ (Unicode letter sequences)"
    )

    print(
        "\nFor your Methods section:\n"
        f"  Tokenizer : {tokenizer_desc}\n"
        "  Lowercase : yes\n"
        "  Length    : Unicode character count\n"
        "  Excluded  : digits, punctuation, spaces\n"
        "  Corpus    : DCEP OQ + QT subdirs; REPORT excluded (absent in some languages)\n"
        f"  Min freq  : {args.min_freq}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
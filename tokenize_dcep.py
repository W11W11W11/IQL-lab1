#!/usr/bin/env python3
"""
Tokenize extracted DCEP-2013 plain-text files and produce word-frequency-length tables.

Reads per-language file lists produced by analyse_index.py (--save-paths).
Each filelist contains paths like:
    xml/EL/RULES-EP/6737258__RULES-EP__20040501__EL.xml.gz

These are translated to actual .txt paths under --input-root, e.g.:
    <input-root>/EL/RULES-EP/6737258__RULES-EP__20040501__EL.txt

Output (one file per language, saved to --out-dir):
    <iso>.txt  with three tab-separated columns:
        word_form  frequency  length

Usage
-----
    # Tokenize all languages whose filelist exists in parallel_selection/:
    python tokenize_dcep.py \\
        --filelists parallel_selection/ \\
        --input-root /path/to/dcep/texts/ \\
        --out-dir data/

    # Fastest: regex tokenizer for all languages (no spaCy needed)
    python tokenize_dcep.py \\
        --filelists parallel_selection/ \\
        --input-root /path/to/dcep/texts/ \\
        --no-spacy

Requirements
------------
    pip install spacy tqdm
    python -m spacy download en_core_web_sm   # only if using spaCy
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
# File resolution from filelists
#
# Filelist lines look like:
#   xml/EL/RULES-EP/6737258__RULES-EP__20040501__EL.xml.gz
#
# We strip the leading "xml/" and the trailing ".xml.gz", then append ".txt",
# giving a relative path:
#   EL/RULES-EP/6737258__RULES-EP__20040501__EL.txt
#
# That is then joined with --input-root to get the absolute path.
# ---------------------------------------------------------------------------

def filelist_path_to_txt(raw: str, input_root: Path) -> Path:
    """Convert an index path entry to the actual .txt file path."""
    # Strip leading "xml/" prefix if present
    if raw.startswith("xml/"):
        raw = raw[4:]
    # Strip .xml.gz and replace with .txt
    if raw.endswith(".xml.gz"):
        raw = raw[:-7] + ".txt"
    elif raw.endswith(".xml"):
        raw = raw[:-4] + ".txt"
    return input_root / raw


def load_filelists(filelists_dir: Path, input_root: Path) -> dict[str, list[Path]]:
    """
    Read all filelist_<LANG>.txt files from filelists_dir.
    Returns {lang -> [resolved .txt Paths]}, skipping files that don't exist.
    """
    lang_map: dict[str, list[Path]] = {}

    filelist_files = sorted(filelists_dir.glob("filelist_*.txt"))
    if not filelist_files:
        print(f"[error] No filelist_*.txt files found in {filelists_dir}", file=sys.stderr)
        return lang_map

    for fl in filelist_files:
        # Extract lang code from filename: filelist_EL.txt -> EL
        m = re.match(r"filelist_([A-Z]{2})\.txt$", fl.name, re.IGNORECASE)
        if not m:
            continue
        lang = m.group(1).upper()

        resolved: list[Path] = []
        missing = 0
        for line in fl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            txt_path = filelist_path_to_txt(line, input_root)
            if txt_path.exists():
                resolved.append(txt_path)
            else:
                missing += 1

        if missing > 0:
            print(f"  [warn] {lang}: {missing} paths from filelist not found on disk",
                  file=sys.stderr)
        if resolved:
            lang_map[lang] = resolved
        else:
            print(f"  [skip] {lang}: no files resolved from filelist", file=sys.stderr)

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
        description="Tokenize DCEP parallel .txt files -> word/frequency/length tables."
    )
    p.add_argument("--filelists", required=True,
                   help="Directory containing filelist_<LANG>.txt files "
                        "(output of analyse_index.py --save-paths).")
    p.add_argument("--input-root", required=True,
                   help="Root directory where the actual .txt files live. "
                        "Filelist paths are resolved relative to this.")
    p.add_argument("--out-dir", default="data",
                   help="Output directory for <iso>.txt tables (default: data/).")
    p.add_argument("--langs", nargs="*", default=None,
                   help="Only process these language codes (default: all filelists found).")
    p.add_argument("--min-freq", type=int, default=1,
                   help="Min token frequency to include (default: 1).")
    p.add_argument("--no-spacy", action="store_true",
                   help="Use fast regex tokenizer for ALL languages (no spaCy). "
                        "Fastest option — document in Methods if used.")
    args = p.parse_args()

    filelists_dir = Path(args.filelists)
    input_root    = Path(args.input_root)
    out_dir       = Path(args.out_dir)

    if not filelists_dir.is_dir():
        print(f"[error] --filelists directory not found: {filelists_dir}", file=sys.stderr)
        return 1
    if not input_root.is_dir():
        print(f"[error] --input-root directory not found: {input_root}", file=sys.stderr)
        return 1

    lang_map = load_filelists(filelists_dir, input_root)
    if not lang_map:
        print("[error] No languages could be loaded from filelists.", file=sys.stderr)
        return 1

    if args.langs:
        requested = {l.upper() for l in args.langs}
        missing = requested - set(lang_map)
        if missing:
            print(f"[warn] Requested but no filelist found for: {sorted(missing)}", file=sys.stderr)
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
        "spaCy (tokenizer only, no pipeline) with regex fallback for HU, (others without model)"
        if use_spacy
        else "regex [^\\W\\d_]+ (Unicode letter sequences, all languages)"
    )

    print("\n=== All done ===")
    print(
        "\nFor your Methods section:\n"
        f"  Tokenizer  : {tokenizer_desc}\n"
        "  Lowercase  : yes\n"
        "  Length     : Unicode character count\n"
        "  Excluded   : digits, punctuation, spaces\n"
        "  Corpus     : parallel subset from DCEP cross-lingual index\n"
        f"  Min freq   : {args.min_freq}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
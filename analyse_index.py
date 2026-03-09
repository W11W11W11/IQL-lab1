#!/usr/bin/env python3
"""
Analyse DCEP cross-lingual-index.txt to find parallel documents and
help decide which languages to use.

Usage
-----
# Show coverage stats for all languages found in the index:
python analyse_index.py --index cross-lingual-index.txt

# Show how many parallel docs exist for a specific set of languages:
python analyse_index.py --index cross-lingual-index.txt --langs EN FR FI HU MT BG

# Write a filtered index keeping only fully-parallel lines for chosen langs:
python analyse_index.py --index cross-lingual-index.txt --langs EN FR FI HU MT BG --save-index

# Write out the parallel file paths per language (one file per lang):
python analyse_index.py --index cross-lingual-index.txt --langs EN FR FI HU MT BG --save-paths
"""

from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path
from itertools import combinations


# ---------------------------------------------------------------------------
# Parse index
# ---------------------------------------------------------------------------

def parse_index(index_path: Path) -> list[dict[str, str]]:
    """
    Parse cross-lingual-index.txt.

    Each non-empty line is a set of parallel files for one document group.
    Returns a list of dicts: {lang_code -> file_path} per document group.
    """
    groups = []
    with index_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries = line.split()
            lang_map: dict[str, str] = {}
            for entry in entries:
                # Extract language code from path like xml/EN/WQA/....xml.gz
                m = re.match(r"xml/([A-Z]{2})/", entry)
                if m:
                    lang = m.group(1)
                    lang_map[lang] = entry
            if lang_map:
                groups.append(lang_map)
    return groups


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def language_coverage(groups: list[dict[str, str]]) -> dict[str, int]:
    """Count how many document groups each language appears in."""
    counts: dict[str, int] = defaultdict(int)
    for g in groups:
        for lang in g:
            counts[lang] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def parallel_count(groups: list[dict[str, str]], langs: set[str]) -> int:
    """Count groups where ALL langs in the set are present."""
    return sum(1 for g in groups if langs.issubset(g.keys()))


def filter_parallel(
    groups: list[dict[str, str]], langs: set[str]
) -> list[dict[str, str]]:
    """Return only groups where ALL langs are present."""
    return [g for g in groups if langs.issubset(g.keys())]


# ---------------------------------------------------------------------------
# Language family info (for display)
# ---------------------------------------------------------------------------

FAMILIES = {
    "BG": "IE / Slavic",
    "CS": "IE / Slavic",
    "DA": "IE / Germanic",
    "DE": "IE / Germanic",
    "EL": "IE / Hellenic",
    "EN": "IE / Germanic",
    "ES": "IE / Romance",
    "ET": "Uralic / Finnic",
    "FI": "Uralic / Finnic",
    "FR": "IE / Romance",
    "GA": "IE / Celtic",
    "HU": "Uralic / Ugric",
    "IT": "IE / Romance",
    "LT": "IE / Baltic",
    "LV": "IE / Baltic",
    "MT": "Afro-Asiatic / Semitic",
    "NL": "IE / Germanic",
    "PL": "IE / Slavic",
    "PT": "IE / Romance",
    "RO": "IE / Romance",
    "SK": "IE / Slavic",
    "SL": "IE / Slavic",
    "SV": "IE / Germanic",
    "TR": "Turkic",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse DCEP cross-lingual index for parallel document coverage."
    )
    parser.add_argument("--index", required=True,
                        help="Path to cross-lingual-index.txt")
    parser.add_argument("--langs", nargs="*", default=None,
                        help="Language codes to analyse (e.g. EN FR FI HU MT BG). "
                             "If omitted, shows stats for all languages.")
    parser.add_argument("--save-index", action="store_true",
                        help="Write filtered index lines (requires --langs).")
    parser.add_argument("--save-paths", action="store_true",
                        help="Write one file-list per language (requires --langs).")
    parser.add_argument("--out-dir", default=".",
                        help="Where to write output files (default: current dir).")
    args = parser.parse_args()

    index_path = Path(args.index)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {index_path} ...")
    groups = parse_index(index_path)
    print(f"  Total document groups in index: {len(groups):,}\n")

    # ------------------------------------------------------------------
    # 1. Per-language coverage table (always shown)
    # ------------------------------------------------------------------
    coverage = language_coverage(groups)
    all_langs = sorted(coverage.keys())

    print("=" * 60)
    print(f"{'Lang':<6} {'Family':<26} {'Doc groups':>10}  {'% of total':>10}")
    print("-" * 60)
    for lang, count in sorted(coverage.items(), key=lambda x: -x[1]):
        family = FAMILIES.get(lang, "Unknown")
        pct    = 100 * count / len(groups)
        print(f"  {lang:<4} {family:<26} {count:>10,}  {pct:>9.1f}%")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 2. Pairwise parallel coverage matrix (helps spot good combinations)
    # ------------------------------------------------------------------
    print("Pairwise parallel document counts (docs present in BOTH languages):")
    print("(Use this to spot languages with high mutual coverage)\n")

    # Only show langs that appear in the index
    matrix_langs = all_langs
    header = "     " + "".join(f"{l:>6}" for l in matrix_langs)
    print(header)
    for la in matrix_langs:
        row = f"{la:<5}"
        for lb in matrix_langs:
            if la == lb:
                row += f"{'--':>6}"
            else:
                n = parallel_count(groups, {la, lb})
                row += f"{n:>6,}"
        print(row)
    print()

    # ------------------------------------------------------------------
    # 3. If --langs given: full analysis for that specific set
    # ------------------------------------------------------------------
    if args.langs:
        chosen = {l.upper() for l in args.langs}
        unknown = chosen - set(all_langs)
        if unknown:
            print(f"[warn] These languages were not found in the index: {unknown}")
            chosen -= unknown

        parallel = filter_parallel(groups, chosen)
        n_parallel = len(parallel)

        print("=" * 60)
        print(f"Chosen languages: {sorted(chosen)}")
        print(f"Fully parallel document groups: {n_parallel:,}")
        print(f"  (= groups where ALL {len(chosen)} languages are present)")
        print("=" * 60)

        if n_parallel == 0:
            print("\n[!] No fully parallel documents for this combination.")
            print("    Try dropping a low-coverage language.")
        else:
            # Show per-language file counts in the parallel subset
            print(f"\nFiles per language in the parallel subset:")
            for lang in sorted(chosen):
                files = [g[lang] for g in parallel]
                print(f"  {lang}: {len(files):,} files")

        # Suggest: what is the best subset of N languages from the index
        # that maximises parallel docs, with at least one non-IE language?
        print("\n--- Top 10 combinations of 6 languages by parallel doc count ---")
        print("(from all languages found in the index)\n")

        # For speed, only consider langs with > 100 docs
        candidate_langs = [l for l, c in coverage.items() if c > 100]
        results = []
        for combo in combinations(candidate_langs, 6):
            n = parallel_count(groups, set(combo))
            families = {FAMILIES.get(l, "?").split("/")[0].strip() for l in combo}
            results.append((n, combo, len(families)))

        # Sort by parallel count, then by family diversity
        results.sort(key=lambda x: (-x[0], -x[2]))
        print(f"{'Langs':<36} {'Families':>10} {'Parallel docs':>15}")
        print("-" * 65)
        for n, combo, n_fam in results[:10]:
            langs_str = " ".join(sorted(combo))
            fam_str   = str(n_fam)
            print(f"  {langs_str:<34} {fam_str:>10} {n:>15,}")
        print()

        # ------------------------------------------------------------------
        # Save outputs if requested
        # ------------------------------------------------------------------
        if args.save_index and parallel:
            out_path = out_dir / "parallel-index-filtered.txt"
            with out_path.open("w", encoding="utf-8") as f:
                for g in parallel:
                    line = " ".join(g[lang] for lang in sorted(chosen) if lang in g)
                    f.write(line + "\n")
            print(f"Filtered index written to: {out_path}")

        if args.save_paths and parallel:
            for lang in sorted(chosen):
                out_path = out_dir / f"filelist_{lang}.txt"
                with out_path.open("w", encoding="utf-8") as f:
                    for g in parallel:
                        f.write(g[lang] + "\n")
                print(f"  File list for {lang}: {out_path} ({len(parallel)} entries)")


if __name__ == "__main__":
    main()
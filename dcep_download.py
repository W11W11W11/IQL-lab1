#!/usr/bin/env python3
"""
Download DCEP-2013 resources for selected languages.

Examples
--------
# Download EN + FR monolingual sentence corpora:
python dcep_download.py --langs EN FR

# Download EN + FR + DE monolingual + all pair alignments + extraction scripts:
python dcep_download.py --langs EN FR DE --pairs --scripts

# Download only specific pairs (still downloads needed monolinguals unless --no-mono):
python dcep_download.py --pair EN-FR --pair DE-EN --scripts
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


BASE = "https://wt-public.emm4u.eu/Resources/DCEP-2013"
SENT_URL = BASE + "/sentences/DCEP-sentence-{lang}-pub.tar.bz2"
PAIR_URL = BASE + "/langpairs/DCEP-{a}-{b}.tar.bz2"
SCRIPTS_URL = BASE + "/DCEP-extract-scripts.tar.bz2"

# Languages shown on the official download page (Option 1). :contentReference[oaicite:4]{index=4}
KNOWN_LANGS = {
    "BG","CS","DA","DE","EL","EN","ES","ET","FI","FR","GA","HU","IT","LT","LV","MT",
    "NL","PL","PT","RO","SK","SL","SV","TR"
}


def filename_from_url(url: str) -> str:
    path = urlsplit(url).path
    return os.path.basename(path.rstrip("/")) or "download.bin"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download(url: str, out_dir: str, retries: int = 5, timeout: int = 60) -> str:
    """
    Download with basic resume support via HTTP Range when a partial file exists.
    """
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename_from_url(url))

    existing = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    headers = {"User-Agent": "dcep_download/1.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    attempt = 0
    while True:
        attempt += 1
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                # If server ignores Range, start over.
                status = getattr(resp, "status", None)
                if existing > 0 and status == 200:
                    existing = 0
                    headers.pop("Range", None)
                    out_path_tmp = out_path + ".tmp"
                    # restart cleanly
                    if os.path.exists(out_path_tmp):
                        os.remove(out_path_tmp)
                    # re-request without Range
                    req2 = Request(url, headers=headers)
                    with urlopen(req2, timeout=timeout) as resp2, open(out_path_tmp, "wb") as f2:
                        _stream_copy(resp2, f2, prefix=os.path.basename(out_path))
                    os.replace(out_path_tmp, out_path)
                    return out_path

                mode = "ab" if existing > 0 else "wb"
                out_path_tmp = out_path + ".tmp"
                # write to tmp, then move into place
                with open(out_path_tmp, mode) as f:
                    _stream_copy(resp, f, prefix=os.path.basename(out_path))
                os.replace(out_path_tmp, out_path)
                return out_path

        except (HTTPError, URLError, TimeoutError) as e:
            if attempt >= retries:
                raise RuntimeError(f"Failed after {retries} attempts: {url}\nLast error: {e}") from e
            sleep_s = min(2 ** attempt, 30)
            print(f"\n[warn] download failed (attempt {attempt}/{retries}) -> {e}\n"
                  f"       retrying in {sleep_s}s: {url}", file=sys.stderr)
            time.sleep(sleep_s)


def _stream_copy(resp, f, prefix: str = "") -> None:
    chunk = 1024 * 1024  # 1 MiB
    downloaded = 0
    last_print = time.time()
    while True:
        buf = resp.read(chunk)
        if not buf:
            break
        f.write(buf)
        downloaded += len(buf)
        now = time.time()
        if now - last_print > 0.5:
            mb = downloaded / (1024 * 1024)
            print(f"\r{prefix}: +{mb:.1f} MiB", end="", flush=True)
            last_print = now
    if downloaded:
        mb = downloaded / (1024 * 1024)
        print(f"\r{prefix}: +{mb:.1f} MiB", flush=True)


def norm_lang(code: str) -> str:
    code = code.strip().upper()
    if len(code) != 2:
        raise argparse.ArgumentTypeError(f"Language code must be 2 letters (got {code!r})")
    return code


def norm_pair(pair: str) -> tuple[str, str]:
    pair = pair.strip().upper().replace("_", "-")
    if "-" not in pair:
        raise argparse.ArgumentTypeError(f"Pair must look like EN-FR (got {pair!r})")
    a, b = pair.split("-", 1)
    a, b = norm_lang(a), norm_lang(b)
    # DCEP expects alphabetical ordering for langpairs. :contentReference[oaicite:5]{index=5}
    return tuple(sorted((a, b)))


def main() -> int:
    p = argparse.ArgumentParser(description="Download DCEP-2013 corpora for selected languages.")
    p.add_argument("--langs", nargs="*", type=norm_lang, default=[],
                   help="Language codes (e.g., EN FR DE).")
    p.add_argument("--pair", action="append", default=[],
                   help="Specific pair(s) like EN-FR. Can be repeated.")
    p.add_argument("--pairs", action="store_true",
                   help="Download all pair alignments among --langs.")
    p.add_argument("--scripts", action="store_true",
                   help="Download DCEP-extract-scripts.tar.bz2.")
    p.add_argument("--no-mono", action="store_true",
                   help="Do not download monolingual sentence corpora.")
    p.add_argument("--out", default="dcep_downloads",
                   help="Output directory (default: dcep_downloads).")
    p.add_argument("--retries", type=int, default=5,
                   help="Retries per file (default: 5).")
    args = p.parse_args()

    # Collect language set from --langs and --pair
    langs = set(args.langs)
    pairs: set[tuple[str, str]] = set()

    for raw in args.pair:
        a, b = norm_pair(raw)
        pairs.add((a, b))
        langs.add(a)
        langs.add(b)

    if args.pairs:
        if len(args.langs) < 2:
            print("[warn] --pairs set but fewer than 2 languages in --langs; nothing to do.", file=sys.stderr)
        for a, b in itertools.combinations(sorted(set(args.langs)), 2):
            pairs.add((a, b))

    # Friendly warning if codes are not on the download page list.
    unknown = sorted([x for x in langs if x not in KNOWN_LANGS])
    if unknown:
        print("[warn] Some language codes are not listed on the official DCEP download page option 1:",
              ", ".join(unknown), file=sys.stderr)

    out_sent = os.path.join(args.out, "sentences")
    out_pairs = os.path.join(args.out, "langpairs")
    out_tools = os.path.join(args.out, "tools")

    # Download monolingual sentence corpora
    if not args.no_mono:
        for lang in sorted(langs):
            url = SENT_URL.format(lang=lang)
            print(f"\n==> Download sentence corpus: {lang} ({url})")
            path = download(url, out_sent, retries=args.retries)
            print(f"Saved: {path}")

    # Download alignment tarballs
    for a, b in sorted(pairs):
        url = PAIR_URL.format(a=a, b=b)
        print(f"\n==> Download alignment: {a}-{b} ({url})")
        path = download(url, out_pairs, retries=args.retries)
        print(f"Saved: {path}")

    # Download scripts
    if args.scripts:
        print(f"\n==> Download extraction scripts ({SCRIPTS_URL})")
        path = download(SCRIPTS_URL, out_tools, retries=args.retries)
        print(f"Saved: {path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
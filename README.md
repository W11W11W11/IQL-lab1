E.g. Download EN + FR sentence corpora: 
`python dcep_download.py --langs EN FR`


E.g. Download EN + FR + DE sentence corpora and all pair alignment tarballs between them: `python dcep_download.py --langs EN FR DE --pairs --scripts`

---- 
To unzip everything in dcep_downloads

```console
cd ./decp_downloads

find . -type f -name "*.tar.bz2" -print0 | while IFS= read -r -d '' file; do
  echo "Extracting $file"
  tar -xjf "$file" -C "$(dirname "$file")"
done

```

Create Virtual Environment
-----
Need to use python 3.11 because of spacy
```
python3.11 -m venv .venv

source .venv/bin/activate
```

Requirements
------------
    pip install spacy
    # Then download each language model you need, e.g.:
    python -m spacy download en_core_web_sm
    python -m spacy download fr_core_news_sm
    python -m spacy download de_core_news_sm
    python -m spacy download fi_core_news_sm   # Finnish  (Uralic)
    python -m spacy download el_core_news_sm   # Greek
    python -m spacy download bg_core_news_sm   # Bulgarian
    # For languages without a spaCy model (HU, MT, GA, ET, LV, LT, SK, SL, CS, HR):
    #   we fall back to a whitespace/punct tokenizer — document this in Methods.



Usage
-----
Tokenize DCEP-2013 plain-text files and produce word-frequency-length tables.

Output (one file per language, saved to --out-dir):
    <iso>.txt  with three tab-separated columns:
        word_form  frequency  length

## Process a single extracted DCEP sentence file:
`python tokenize_dcep.py --input dcep_downloads/sentences/ --out-dir data/`

## Process only specific languages:
`python tokenize_dcep.py --input dcep_downloads/sentences/ --langs EN FR FI HU MT GA BG`


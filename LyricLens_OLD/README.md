# LyricLens ML engine: Stage 2a (data engineering)

New, self-contained folder. It does not read, change or delete any of your existing
project files, datasets or models. Everything is written under `data/` here.

## What this stage does
1. `lyriclens/data/build_catalog.py`: turns the raw Kaggle JSONL (551,443 rows) into a clean
   **canonical catalog** (one row per recording, stable `track_id`, lyrics kept in a separate file,
   every raw row preserved in `variants.parquet`, a validation report with integrity checks).
2. `scripts/download_onion.py`: resumable, checksum-verified download of Music4All-Onion.
3. `scripts/inspect_file.py`: prints the real schema of any data file so we never guess formats.
4. `tests/`: 23 automated tests (identity rules, parsers, build, determinism, downloader, inspector).

Tested on Python 3.13 with pandas 3.1.1, pyarrow 26.0.1, numpy 2.x (Linux sandbox).
Not yet run on your Mac: please do step 2 below and tell me the result.

## What YOU do (about 15 minutes of work, mostly waiting)
Run everything from inside this folder.

```bash
# 1. environment (a private venv; your global Python is untouched)
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-ml.txt

# 2. run the tests. Expected last line: "23 passed"
python3 -m pytest -q

# 3. build the catalog from YOUR copy of the big file (use your real path; keep the quotes)
python3 -m lyriclens.data.build_catalog \
    --input "/path/to/final_milliondataset_BERT_500K_revised.json" \
    --out data/processed/catalog
# Expected on your file (this is what my run printed):
#   rows_read 551443 | level1_distinct_raw_artist_title_pairs 498052 | level2_canonical_tracks 491203
#   "all_passed": true        (about 2 to 5 minutes, about 420 MB written)

# 4. download Music4All-Onion (needs about 4 GB free; safe to stop and re-run, it resumes)
python3 scripts/download_onion.py --list                          # shows the real file names
python3 scripts/download_onion.py --preset core+time --out data/raw/onion

# 5. look at each downloaded file once, and send me the output
python3 scripts/inspect_file.py data/raw/onion/<file name>        # repeat for every file
```

## What to send me back
* The last lines of steps 2 and 3 (or any error text, copy-pasted).
* The output of `--list` and of `inspect_file.py` for every Onion file.
* Your MacBook's RAM (8 / 16 / 24 GB). It decides how I size the training jobs.

## Outputs of step 3 (`data/processed/catalog/`)
| file | content |
|---|---|
| `catalog.parquet` | 491,203 tracks: metadata, 7 audio features, tags, emotion, genres, ISRC, no lyrics |
| `lyrics.parquet` | `track_id` + lyrics (separate on purpose; never commit it to a public repo) |
| `variants.parquet` | all 551,443 raw rows mapped to `track_id` (nothing is lost) |
| `similar_edges.parquet` | the dataset's own "similar songs" links: a weak signal, not ground truth |
| `catalog_report.json` | counts, null rates, ranges, ISRC stats, integrity checks |

`docs/stage2a_reference_report.json` is the report from my run on your file, for comparison.

## Identity rules (why the numbers are what they are)
* Same recording = same artist and title after ignoring case, Latin accents, punctuation, spacing.
* "(Live)", "- Remastered 2009" etc. stay separate tracks, but share a `work_key`, so the train/test split
  can keep every version of a song on one side (leakage prevention).
* Non-Latin titles are preserved. If either side of the key is empty, the raw text is hashed in, so
  unrelated songs are never merged.
* Data findings worth knowing: 5,781 "similar song" links point at the song itself; 79 rows had an
  empty artist or title; 23 rows carried junk emotion labels (mapped to `unknown`).

"""
load_nslkdd.py - read NSL-KDD files safely.

Design rules (from the project rules):
  * The original files are only ever OPENED FOR READING. Nothing here writes
    to, renames, or "cleans" a dataset file.
  * The loader does not repair data. If a file does not look like NSL-KDD it
    raises an error instead of guessing.
  * The loader does not drop, encode, scale or split anything. That belongs to
    later phases (preprocessing, split) so each step can be checked alone.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd

from .schema import (
    CATEGORICAL, LABEL_COL, N_RAW_COLUMNS, NORMAL_LABEL, RAW_COLUMNS,
)


def sha256_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 fingerprint of a file. Used to prove a file was not modified."""
    h = hashlib.sha256()
    with open(path, "rb") as f:                       # "rb" = read-only
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def inspect_raw_text(path: str | Path) -> dict:
    """
    Look at the file as plain text, independently of pandas.

    Why: pandas can silently "help" (skip blank lines, shift columns when a
    row has extra fields). Counting fields line-by-line ourselves catches that.
    """
    data = Path(path).read_bytes()
    is_ascii = data.isascii()
    text = data.decode("utf-8", errors="replace")
    ends_with_newline = data.endswith(b"\n")
    lines = text.split("\n")
    if ends_with_newline:
        lines = lines[:-1]                            # drop the empty tail

    n_blank = sum(1 for ln in lines if ln.strip() == "")
    field_hist: Counter = Counter()
    for ln in lines:
        if ln.strip() == "":
            continue
        field_hist[len(next(csv.reader([ln])))] += 1

    first_fields = next(csv.reader([lines[0]])) if lines else []
    # Header heuristic: a data row starts with a number (duration) and has a
    # non-numeric second-to-last field that is an attack name. A header row
    # would have a non-numeric first field.
    header_suspected = (not first_fields) or (not _is_number(first_fields[0]))

    return {
        "n_lines": len(lines),
        "n_blank_lines": n_blank,
        "n_carriage_returns": data.count(b"\r"),
        "is_ascii": is_ascii,
        "ends_with_newline": ends_with_newline,
        "field_count_histogram": dict(field_hist),
        "first_line_fields": first_fields,
        "header_suspected": header_suspected,
    }


def load_nslkdd(path: str | Path) -> pd.DataFrame:
    """
    Load one NSL-KDD file into a DataFrame with the 43 named columns.

    Raises ValueError if the file is not 43 fields on every line, or if a
    column that should be numeric is not. Returns the data exactly as stored.
    """
    path = Path(path)
    raw = inspect_raw_text(path)
    if raw["header_suspected"]:
        raise ValueError(f"{path}: first line looks like a header; this loader "
                         "expects a header-less file. Not guessing - stop.")
    if set(raw["field_count_histogram"]) != {N_RAW_COLUMNS}:
        raise ValueError(f"{path}: expected exactly {N_RAW_COLUMNS} fields on "
                         f"every line, found {raw['field_count_histogram']}")

    string_cols = CATEGORICAL + [LABEL_COL]
    df = pd.read_csv(
        path, header=None, names=RAW_COLUMNS,
        dtype={c: "string" for c in string_cols},
    )

    bad = [c for c in RAW_COLUMNS
           if c not in string_cols and not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"{path}: columns expected numeric but are not: {bad}")
    return df


def binary_label(df: pd.DataFrame) -> pd.Series:
    """normal -> 0, any attack -> 1 (positive class = attack, per the paper)."""
    return (df[LABEL_COL] != NORMAL_LABEL).astype("int8")

"""
audit_nslkdd.py - Phase 1 data audit (NO modelling, NO training).

Run from the project root:
    python -m src.data.audit_nslkdd --train data/raw/KDDTrain_.txt \
                                    --test  data/raw/KDDTest_.txt

What it does
  1. Checks each file's integrity (header? field counts? encoding?).
  2. Checks the 41 intended model inputs, the label and the difficulty column.
  3. Looks for missing / malformed / duplicate / conflicting rows.
  4. Compares KDDTrain+ against every number the paper/spec states.
  5. Describes KDDTest+ STRUCTURE only (see the "KDDTest+ usage" note below).

KDDTest+ usage note (project rule: keep KDDTest+ untouched during development)
  The audit reads KDDTest+ only for descriptive/structural facts: line format,
  label names/counts, category names, and which rows repeat KDDTrain+ rows.
  It computes NO feature statistics from KDDTest+ (no means, no scalers, no
  encoders). Nothing it prints may be used to tune a model or pick features.

Outputs: reports/phase1_audit.txt (this console text) and
         reports/phase1_audit.json (machine-readable facts).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .load_nslkdd import binary_label, inspect_raw_text, load_nslkdd, sha256_of_file
from .schema import (
    BINARY_LIKE_COLUMNS, CANONICAL_INPUTS, CATEGORICAL,
    EXPECTED_CONSTANT_COLUMNS, LABEL_COL, LEVEL_COL, NORMAL_LABEL,
    RATE_COLUMNS, RAW_FEATURES, SPEC_EXPECTED,
)


# ----------------------------------------------------------------- reporting
class Report:
    """Collects console lines, PASS/WARN/FAIL checks and machine-readable facts."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.checks: list[dict] = []
        self.facts: dict = {}

    def say(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.say()
        self.say("=" * 78)
        self.say(title)
        self.say("=" * 78)

    def check(self, desc: str, ok: bool, detail: str = "", fail_level: str = "FAIL") -> None:
        cid = f"P1-{len(self.checks) + 1:02d}"
        status = "PASS" if ok else fail_level
        self.checks.append({"id": cid, "status": status, "check": desc, "detail": detail})
        self.say(f"[{status}] {cid} {desc}" + (f"  -> {detail}" if detail else ""))

    def info(self, text: str) -> None:
        self.say(f"[INFO] {text}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return sorted(o) if isinstance(o, set) else list(o)
    return str(o)


def _fmt_counts(counts: pd.Series, width: int = 18) -> list[str]:
    return [f"    {str(k):<{width}} {int(v):>8,}" for k, v in counts.items()]


def feature_row_hashes(df: pd.DataFrame) -> np.ndarray:
    """One 64-bit fingerprint per row, computed from the 41 raw feature columns
    only (not label, not level). Used to find repeated feature vectors."""
    return pd.util.hash_pandas_object(df[RAW_FEATURES], index=False).to_numpy()


# ---------------------------------------------------- checks common to both files
def audit_one_file(rep: Report, tag: str, path: Path, is_train: bool) -> pd.DataFrame:
    rep.section(f"{tag}: file integrity and format  [{path}]")
    if not path.exists():
        rep.say(f"ERROR: file not found: {path}")
        sys.exit(2)

    sha_before = sha256_of_file(path)
    raw = inspect_raw_text(path)
    rep.say(f"  size           : {path.stat().st_size:,} bytes")
    rep.say(f"  sha256         : {sha_before}")
    rep.say(f"  first line     : {','.join(raw['first_line_fields'][:6])},...")
    rep.check("No header row (first field is numeric)", not raw["header_suspected"])
    rep.check("Every line has exactly 43 fields",
              set(raw["field_count_histogram"]) == {43},
              f"field-count histogram = {raw['field_count_histogram']}")
    rep.check("No blank lines", raw["n_blank_lines"] == 0, f"{raw['n_blank_lines']} blank")
    rep.check("Plain ASCII text", raw["is_ascii"])
    if raw["n_carriage_returns"]:
        rep.info(f"file contains {raw['n_carriage_returns']:,} carriage returns (Windows line endings)")

    df = load_nslkdd(path)
    rep.facts[tag] = {"path": str(path), "sha256": sha_before, "bytes": path.stat().st_size,
                      "rows": int(len(df)), "cols": int(df.shape[1]), "raw_text": raw}
    rep.say(f"  rows x columns : {len(df):,} x {df.shape[1]}")
    n_num = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    rep.say(f"  column types   : {n_num} numeric, {df.shape[1] - n_num} text "
            f"(text = {CATEGORICAL + [LABEL_COL]})")

    # ---- feature set --------------------------------------------------
    rep.section(f"{tag}: the 41 intended model inputs, label and difficulty column")
    constant = [c for c in RAW_FEATURES if df[c].nunique(dropna=False) == 1]
    rep.say(f"  constant raw-feature columns here: {constant}")
    if is_train:
        # The decision to drop a column is made from TRAIN data only.
        inputs = [c for c in RAW_FEATURES if c not in constant] + [LEVEL_COL]
        rep.check("Constant column(s) == spec expectation ['num_outbound_cmds']",
                  constant == EXPECTED_CONSTANT_COLUMNS, f"found {constant}")
        rep.check("40 varying raw features + 'level' == the spec's 41 inputs "
                  "(same names, same order)", inputs == CANONICAL_INPUTS,
                  f"{len(inputs)} derived inputs")
        rep.facts[tag]["derived_inputs"] = inputs
    else:
        rep.check("'num_outbound_cmds' is also constant in this file (consistency only)",
                  constant == EXPECTED_CONSTANT_COLUMNS, f"found {constant}")
    rep.say(f"  label column   : '{LABEL_COL}'  ({df[LABEL_COL].nunique()} distinct values)")
    lv = df[LEVEL_COL]
    rep.say(f"  difficulty col : '{LEVEL_COL}'  integer range {int(lv.min())}..{int(lv.max())}")

    # ---- missing / malformed -----------------------------------------
    rep.section(f"{tag}: missing values and malformed values")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    n_nan = int(df.isna().sum().sum())
    n_inf = int(np.isinf(df[num_cols].to_numpy(dtype=float)).sum())
    rep.check("No missing values", n_nan == 0, f"{n_nan} missing cells")
    rep.check("No infinite values", n_inf == 0, f"{n_inf} inf cells")
    neg = {c: int((df[c] < 0).sum()) for c in num_cols if (df[c] < 0).any()}
    rep.check("No negative numbers in any numeric column", not neg, f"{neg}" if neg else "")
    rate_bad = {c: int(((df[c] < 0) | (df[c] > 1)).sum()) for c in RATE_COLUMNS
                if ((df[c] < 0) | (df[c] > 1)).any()}
    rep.check(f"All {len(RATE_COLUMNS)} '*_rate' columns lie in [0, 1]", not rate_bad,
              f"{rate_bad}" if rate_bad else "")
    nonbin = {c: sorted(int(v) for v in set(df[c].unique()) - {0, 1}) for c in BINARY_LIKE_COLUMNS
              if set(df[c].unique()) - {0, 1}}
    rep.check("Binary-looking columns only contain 0/1", not nonbin,
              f"extra values found: {nonbin}" if nonbin else "", fail_level="WARN")
    str_cols = CATEGORICAL + [LABEL_COL]
    ws = {c: int((df[c] != df[c].str.strip()).sum()) for c in str_cols}
    rep.check("No stray whitespace in text columns", not any(ws.values()), f"{ws}" if any(ws.values()) else "")
    dots = int(df[LABEL_COL].str.endswith(".").sum())
    rep.check("No trailing '.' on label names (as in raw KDD'99)", dots == 0, f"{dots} rows")

    # ---- categorical columns -----------------------------------------
    rep.section(f"{tag}: categorical columns")
    cat_facts = {}
    for c in CATEGORICAL:
        vals = sorted(df[c].unique())
        cat_facts[c] = vals
        shown = vals if len(vals) <= 12 else f"{vals[:6]} ... ({len(vals)} distinct)"
        rep.say(f"  {c:<14}: {len(vals):>3} distinct  {shown}")
    rep.facts[tag]["categorical_values"] = cat_facts

    # ---- duplicates ---------------------------------------------------
    rep.section(f"{tag}: duplicate and conflicting rows")
    n_full = int(df.duplicated().sum())
    rep.check("No fully identical rows (all 43 fields)", n_full == 0, f"{n_full} duplicates",
              fail_level="WARN")
    h = feature_row_hashes(df)
    n_feat_dup = int(pd.Series(h).duplicated().sum())
    grp = pd.DataFrame({"h": h, "label": df[LABEL_COL]}).groupby("h")["label"].nunique()
    n_conflict = int((grp > 1).sum())
    rep.say(f"  rows whose 41-feature vector repeats an earlier row: {n_feat_dup:,}")
    rep.say(f"  feature vectors that appear with MORE THAN ONE label : {n_conflict:,}")
    rep.info("Identical features with different labels cannot both be classified correctly "
             "by any model - this is irreducible noise, not a bug in our code.")
    rep.facts[tag].update({"full_row_duplicates": n_full, "feature_vector_repeats": n_feat_dup,
                           "conflicting_label_groups": n_conflict})
    sha_after = sha256_of_file(path)
    rep.facts[tag]["sha256_after_audit"] = sha_after
    rep.check("File bytes unchanged by the audit (sha256 before == after)", sha_before == sha_after)
    return df


# ------------------------------------------------------------ train-only checks
def audit_train_labels(rep: Report, df: pd.DataFrame) -> dict:
    exp = SPEC_EXPECTED
    rep.section("KDDTrain+: labels vs paper / specification")
    counts = df[LABEL_COL].value_counts()
    n_normal = int(counts.get(NORMAL_LABEL, 0))
    n_attack = int(len(df) - n_normal)
    attack_counts = counts.drop(NORMAL_LABEL)
    rep.say("  class counts (all classes, descending):")
    for ln in _fmt_counts(counts):
        rep.say(ln)

    rep.check(f"Rows == {exp['train_rows']:,}", len(df) == exp["train_rows"], f"observed {len(df):,}")
    rep.check(f"Normal == {exp['train_normal']:,}", n_normal == exp["train_normal"], f"observed {n_normal:,}")
    rep.check(f"Attack == {exp['train_attack']:,}", n_attack == exp["train_attack"], f"observed {n_attack:,}")
    rep.check(f"Attack classes == {exp['train_attack_classes']}",
              len(attack_counts) == exp["train_attack_classes"], f"observed {len(attack_counts)}")

    zd_exp = exp["zero_day_counts"]
    zd_obs = {c: int(attack_counts.get(c, 0)) for c in zd_exp}
    rep.check("The 8 zero-day classes have exactly the counts in paper Fig. 4",
              zd_obs == zd_exp, f"observed {zd_obs}")
    oc_obs = {c: int(attack_counts.get(c, 0)) for c in exp["other_counts"]}
    rep.check("Other counts quoted in the text match "
              "(guess_passwd 53, buffer_overflow 30, warezmaster 20, land 18)",
              oc_obs == exp["other_counts"], f"observed {oc_obs}")

    # rare-class rule (Algorithm 2, step 4), swept over delta
    zd_set = set(zd_exp)
    works = [d for d in range(0, 61)
             if {c for c, n in attack_counts.items() if n <= d} == zd_set]
    rep.check("Rare-class rule n(C) <= delta selects exactly the 8 spec classes at delta=14",
              {c for c, n in attack_counts.items() if n <= exp["delta"]} == zd_set)
    rep.say(f"  delta values (0..60) that select exactly those 8 classes: "
            f"{min(works)}..{max(works)}" if works else "  no delta selects exactly those 8 classes")
    rep.check("That range is 11..17 as the spec claims (AT-02)",
              bool(works) and (min(works), max(works)) == (11, 17), f"observed {(min(works), max(works)) if works else None}")

    known_data = sorted(set(attack_counts.index) - zd_set)
    rep.check("The 14 remaining ('known') attack classes equal the paper's Fig. 5 list",
              known_data == sorted(exp["known_attacks"]), f"{len(known_data)} classes")

    # ---- expected split arithmetic (NOT a split; just arithmetic) ----------
    rep.section("KDDTrain+: what the 60/20/20 + rare-holdout protocol SHOULD give (arithmetic only)")
    n_zd = sum(zd_obs.values())
    n_known_attack = n_attack - n_zd
    r_tr, r_va, r_te = exp["split_ratio"]
    rep.say(f"  zero-day rows moved to test only : {n_zd}")
    rep.say(f"  rows left to split               : {len(df) - n_zd:,} "
            f"({n_normal:,} normal + {n_known_attack:,} known attack)")
    rows = [("normal", n_normal, exp["fig3"]["normal"], 0),
            ("attack", n_known_attack, exp["fig3"]["attack"], n_zd)]
    rep.say(f"  {'':<8}{'ideal train':>12}{'ideal val':>11}{'ideal test':>12}   | paper Fig. 3 (train/val/test)")
    for name, n, fig3, extra in rows:
        rep.say(f"  {name:<8}{r_tr * n:>12,.1f}{r_va * n:>11,.1f}{r_te * n + extra:>12,.1f}   | "
                f"{fig3['train']:,} / {fig3['val']:,} / {fig3['test']:,}")
    gap = r_tr * n_known_attack - exp["fig3"]["attack"]["train"]
    rep.check("Fig. 3 TRAIN-attack count (33,558) is consistent with 60% of known attacks",
              abs(gap) < 5, f"60% of {n_known_attack:,} = {r_tr * n_known_attack:,.1f}; "
              f"paper shows 33,558; gap = {gap:,.0f} rows (spec contradiction C-5; "
              f"all other Fig. 3 numbers match within rounding)", fail_level="WARN")
    rep.info("Per-stratum rounding in Phase 2 may shift counts by +-1; a gap of ~1,588 is not rounding.")

    # ---- difficulty column vs label (descriptive, TRAIN only) --------------
    rep.section("KDDTrain+: 'level' (difficulty score) vs the binary label  [relevant to spec D-2]")
    y = binary_label(df)
    stats = df.groupby(y)[LEVEL_COL].agg(["mean", "median", "min", "max"])
    for lab, name in ((0, "normal"), (1, "attack")):
        r = stats.loc[lab]
        rep.say(f"  {name:<7} level mean={r['mean']:.2f}  median={r['median']:.0f}  "
                f"min={int(r['min'])}  max={int(r['max'])}")
    rep.info("'level' is a score the dataset creators computed about each record. The paper feeds it to the "
             "model as a feature (spec D-2: we will run with and without it).")

    return {"n_normal": n_normal, "n_attack": n_attack, "zero_day_observed": zd_obs,
            "known_attack_classes": known_data, "delta_range_ok": [min(works), max(works)] if works else None}


# ------------------------------------------------------- KDDTest+ structure
def audit_test_structure(rep: Report, tr: pd.DataFrame, te: pd.DataFrame, delta: int) -> None:
    rep.section("KDDTest+: STRUCTURE only (labels, categories, row overlap). No model use.")
    tr_counts = tr[LABEL_COL].value_counts()
    rare = {c for c, n in tr_counts.items() if c != NORMAL_LABEL and n <= delta}
    train_labels = set(tr_counts.index)
    known = train_labels - rare - {NORMAL_LABEL}

    te_counts = te[LABEL_COL].value_counts()
    n_normal = int(te_counts.get(NORMAL_LABEL, 0))
    rep.say(f"  rows: {len(te):,}   normal: {n_normal:,}   attack: {len(te) - n_normal:,}   "
            f"distinct labels: {te[LABEL_COL].nunique()}")

    # Partition every test label using ONLY the KDDTrain+ class list.
    def group(lbl: str) -> str:
        if lbl == NORMAL_LABEL:
            return "normal"
        if lbl in known:
            return "known (seen in KDDTrain+, kept for training)"
        if lbl in rare:
            return "rare-8 (in KDDTrain+ but held out by the paper's rule)"
        return "novel (label never appears in KDDTrain+)"

    grp = te[LABEL_COL].map(group)
    rep.say("  KDDTest+ rows by zero-day category (category decided from KDDTrain+ class names only):")
    for g, n in grp.value_counts().items():
        rep.say(f"    {g:<62}{int(n):>7,}")
    rare_in_test = {c: int(te_counts.get(c, 0)) for c in sorted(rare)}
    novel = te_counts[[c for c in te_counts.index if c not in train_labels]]
    rep.say(f"  rare-8 rows in KDDTest+ per class: {rare_in_test}")
    rep.say(f"  novel classes ({len(novel)}) and rows: " + ", ".join(f"{k}={int(v)}" for k, v in novel.items()))
    absent = sorted(train_labels - set(te_counts.index))
    rep.say(f"  KDDTrain+ classes that never occur in KDDTest+: {absent}")
    rep.check("KDDTest+ contains attack labels that never occur in KDDTrain+ (natural novelty exists)",
              len(novel) > 0, f"{len(novel)} novel classes, {int(novel.sum()):,} rows", fail_level="WARN")

    # category names
    rep.say("  category names in KDDTest+ not present in KDDTrain+ (would need the 'UNK' code, spec D-1):")
    unseen_any = False
    for c in CATEGORICAL:
        un = sorted(set(te[c]) - set(tr[c]))
        unseen_any |= bool(un)
        rep.say(f"    {c:<14}: {un if un else 'none'}  (rows affected: {int(te[c].isin(un).sum())})")
    rep.info("Categories present in KDDTrain+ but absent from KDDTest+: " + "; ".join(
        f"{c}={sorted(set(tr[c]) - set(te[c]))}" for c in CATEGORICAL))

    # row overlap with training file
    h_tr, h_te = feature_row_hashes(tr), feature_row_hashes(te)
    overlap = np.isin(h_te, h_tr)
    n_over = int(overlap.sum())
    rep.say(f"  KDDTest+ rows whose 41-feature vector also occurs in KDDTrain+: {n_over:,} "
            f"({100 * n_over / len(te):.1f}% of KDDTest+)")
    rep.check("No KDDTest+ row repeats a KDDTrain+ feature vector", n_over == 0,
              f"{n_over:,} overlapping rows -> secondary evaluation must report results with and "
              f"without them", fail_level="WARN")
    rep.facts["kddtest_structure"] = {
        "rows_by_category": grp.value_counts().to_dict(), "rare8_rows": rare_in_test,
        "novel_classes": {k: int(v) for k, v in novel.items()}, "train_classes_absent_in_test": absent,
        "rows_overlapping_train_feature_vectors": n_over, "delta_used_to_define_rare": delta,
        "usage_note": "structural/descriptive only; no feature statistics computed from KDDTest+",
    }


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1 NSL-KDD data audit (no training).")
    ap.add_argument("--train", default="data/raw/KDDTrain_.txt")
    ap.add_argument("--test", default="data/raw/KDDTest_.txt")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--strict", action="store_true", help="exit with code 1 if any check FAILs")
    args = ap.parse_args(argv)

    rep = Report()
    rep.say("FAS-XAI-ZDA reproduction - PHASE 1 DATA AUDIT")
    rep.say(f"run at (UTC): {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
    rep.say(f"python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__}")

    tr = audit_one_file(rep, "KDDTrain+", Path(args.train), is_train=True)
    train_summary = audit_train_labels(rep, tr)
    te = audit_one_file(rep, "KDDTest+", Path(args.test), is_train=False)
    audit_test_structure(rep, tr, te, SPEC_EXPECTED["delta"])

    n = {s: sum(c["status"] == s for c in rep.checks) for s in ("PASS", "WARN", "FAIL")}
    rep.section("SUMMARY")
    for c in rep.checks:
        if c["status"] != "PASS":
            rep.say(f"  [{c['status']}] {c['id']} {c['check']}")
    rep.say(f"RESULT: {n['PASS']} PASS, {n['WARN']} WARN, {n['FAIL']} FAIL "
            f"(WARN = a real finding to read, not a crash; FAIL = data disagrees with spec/format)")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase1_audit.txt").write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    payload = {"environment": {"python": platform.python_version(), "pandas": pd.__version__,
                               "numpy": np.__version__, "argv": sys.argv},
               "checks": rep.checks, "train_summary": train_summary, "files": rep.facts}
    (out / "phase1_audit.json").write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    rep.say(f"\nWrote {out / 'phase1_audit.txt'} and {out / 'phase1_audit.json'}")
    return 1 if (args.strict and n["FAIL"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

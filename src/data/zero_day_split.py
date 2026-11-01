"""Create the paper-inspired primary zero-day split from KDDTrain+ only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .load_nslkdd import load_nslkdd
from .schema import LABEL_COL, LEVEL_COL, NORMAL_LABEL, SPEC_EXPECTED, RAW_FEATURES

RARE_CLASSES = set(SPEC_EXPECTED["zero_day_counts"])
KNOWN_ATTACKS = set(SPEC_EXPECTED["known_attacks"])
FEATURES_WITH_LEVEL = [c for c in RAW_FEATURES if c != "num_outbound_cmds"] + [LEVEL_COL]

def make_split(train_path: str | Path, out_dir: str | Path, seed: int = 42):
    df = load_nslkdd(train_path)
    observed = set(df[LABEL_COL].unique())
    expected = {NORMAL_LABEL} | KNOWN_ATTACKS | RARE_CLASSES
    if observed != expected:
        raise ValueError(f"Unexpected training labels. Missing={expected-observed}; extra={observed-expected}")
    rare = df[df[LABEL_COL].isin(RARE_CLASSES)].copy()
    known_pool = df[~df[LABEL_COL].isin(RARE_CLASSES)].copy()
    train_df, remainder = train_test_split(
        known_pool, test_size=0.4, random_state=seed, stratify=known_pool[LABEL_COL]
    )
    val_df, known_test_df = train_test_split(
        remainder, test_size=0.5, random_state=seed, stratify=remainder[LABEL_COL]
    )
    test_df = pd.concat([known_test_df, rare], ignore_index=True)
    # Stable shuffle so all rare examples are present but not grouped at the end.
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out/'train.csv', index=False)
    val_df.to_csv(out/'validation.csv', index=False)
    test_df.to_csv(out/'test_primary.csv', index=False)
    summary = {
        "seed": seed, "features_with_level": FEATURES_WITH_LEVEL,
        "split_rows": {"train": len(train_df), "validation": len(val_df), "test_primary": len(test_df)},
        "split_label_counts": {
            name: frame[LABEL_COL].value_counts().sort_index().to_dict()
            for name, frame in [("train",train_df),("validation",val_df),("test_primary",test_df)]
        },
        "rare_classes_appended_to_test": sorted(RARE_CLASSES),
        "rare_rows_appended": int(len(rare)),
        "note": "KDDTest+ is not used in this primary split or for fitting preprocessing."
    }
    (out/'split_summary.json').write_text(json.dumps(summary, indent=2))
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train', default='data/raw/KDDTrain_.txt')
    p.add_argument('--out', default='data/processed/primary_split')
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args()
    s=make_split(a.train,a.out,a.seed)
    print(json.dumps({k:v for k,v in s.items() if k not in ('split_label_counts','features_with_level')}, indent=2))
    print("Wrote split summary and CSVs to", a.out)
if __name__=='__main__': main()

"""Train-only encoding/scaling for the primary FAS-XAI-ZDA experiment."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .schema import CATEGORICAL, LABEL_COL, LEVEL_COL, NORMAL_LABEL, RAW_FEATURES

FEATURES = [c for c in RAW_FEATURES if c != "num_outbound_cmds"] + [LEVEL_COL]
UNK = "__UNK__"

def fit_transform_splits(split_dir: str | Path, out_dir: str | Path, include_level: bool=True):
    split_dir, out_dir = Path(split_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names=['train','validation','test_primary']
    frames={n:pd.read_csv(split_dir/f'{n}.csv') for n in names}
    features=FEATURES.copy()
    if not include_level: features.remove(LEVEL_COL)
    # Fit category vocabularies using the training split only.
    encoders={}
    for col in CATEGORICAL:
        le=LabelEncoder()
        train_values=frames['train'][col].astype(str)
        le.fit(sorted(set(train_values.tolist()) | {UNK}))
        encoders[col]=le
    encoded={}
    for name, frame in frames.items():
        X=frame[features].copy()
        for col, le in encoders.items():
            values=frame[col].astype(str)
            known=set(le.classes_)
            X[col]=values.map(lambda x: x if x in known else UNK)
            X[col]=le.transform(X[col])
        y=(frame[LABEL_COL].astype(str)!=NORMAL_LABEL).astype('int8').to_numpy()
        encoded[name]=(X.astype('float64'),y,frame[LABEL_COL].astype(str).to_numpy())
    scaler=StandardScaler()
    X_train=scaler.fit_transform(encoded['train'][0])
    for name in names:
        X,y,labels=encoded[name]
        arr=scaler.transform(X) if name!='train' else X_train
        # Keras recurrent input: batch x feature-position x one channel.
        np.savez_compressed(out_dir/f'{name}.npz', X=arr.astype('float32')[...,None],
                            y=y.astype('int8'), labels=labels, feature_names=np.array(features))
    bundle={'features':features,'categorical':CATEGORICAL,'encoders':encoders,'scaler':scaler,
            'include_level':include_level,'unknown_token':UNK}
    joblib.dump(bundle,out_dir/'preprocessor.joblib')
    meta={'features':features,'include_level':include_level,'fit_on':'train split only',
          'train_rows':len(frames['train']), 'validation_rows':len(frames['validation']),
          'test_primary_rows':len(frames['test_primary']),
          'shape_convention':'(samples, number_of_features, 1)',
          'binary_labels':{'normal':0,'attack':1}}
    (out_dir/'preprocessing_summary.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
    return meta

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split-dir',default='data/processed/primary_split')
    p.add_argument('--out',default='data/processed/arrays_with_level')
    p.add_argument('--without-level',action='store_true')
    a=p.parse_args()
    fit_transform_splits(a.split_dir,a.out,not a.without_level)
if __name__=='__main__': main()

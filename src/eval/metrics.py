"""Evaluation helpers; positive class is attack (1)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score,
                             classification_report, balanced_accuracy_score)

def calculate_metrics(y_true, probabilities, threshold=0.5):
    y_true=np.asarray(y_true).astype(int)
    probabilities=np.asarray(probabilities).reshape(-1)
    pred=(probabilities>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_true,pred,labels=[0,1]).ravel()
    return {
      'threshold':float(threshold),'n_samples':int(len(y_true)),
      'accuracy':float(accuracy_score(y_true,pred)),
      'balanced_accuracy':float(balanced_accuracy_score(y_true,pred)),
      'precision_attack':float(precision_score(y_true,pred,zero_division=0)),
      'recall_attack':float(recall_score(y_true,pred,zero_division=0)),
      'f1_attack':float(f1_score(y_true,pred,zero_division=0)),
      'roc_auc':float(roc_auc_score(y_true,probabilities)) if len(np.unique(y_true))==2 else None,
      'average_precision':float(average_precision_score(y_true,probabilities)) if len(np.unique(y_true))==2 else None,
      'confusion_matrix':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},
      'classification_report':classification_report(y_true,pred,labels=[0,1],target_names=['normal','attack'],output_dict=True,zero_division=0)
    }

"""Train and evaluate the primary rare-class holdout experiment."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.models.fas_bilstm import build_model
from src.eval.metrics import calculate_metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arrays',default='data/processed/arrays_with_level')
    p.add_argument('--out',default='reports/primary_with_level')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)
    arrays=Path(a.arrays); out=Path(a.out); (out/'plots').mkdir(parents=True,exist_ok=True)
    tr=np.load(arrays/'train.npz',allow_pickle=False)
    va=np.load(arrays/'validation.npz',allow_pickle=False)
    te=np.load(arrays/'test_primary.npz',allow_pickle=False)
    Xtr,ytr=tr['X'],tr['y']; Xva,yva=va['X'],va['y']; Xte,yte=te['X'],te['y']
    features=tr['feature_names'].tolist()
    model,attention_model=build_model(Xtr.shape[1])
    (out/'model_summary.txt').write_text('')
    with (out/'model_summary.txt').open('w') as f: model.summary(print_fn=lambda s:f.write(s+'\n'))
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True),
               tf.keras.callbacks.ModelCheckpoint(str(out/'best_model.keras'),monitor='val_loss',save_best_only=True)]
    hist=model.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=a.epochs,batch_size=a.batch_size,
                   callbacks=callbacks,verbose=2,shuffle=True)
    model.save(out/'final_model.keras')
    prob=model.predict(Xte,batch_size=a.batch_size,verbose=0).reshape(-1)
    metrics=calculate_metrics(yte,prob)
    metrics['split']='primary rare-class holdout from KDDTrain+'
    metrics['feature_count']=len(features); metrics['feature_names']=features
    metrics['epochs_requested']=a.epochs; metrics['epochs_ran']=len(hist.history['loss'])
    (out/'test_metrics.json').write_text(json.dumps(metrics,indent=2))
    # Training curves
    for metric in ['loss','accuracy','auc']:
        if metric in hist.history:
            plt.figure(); plt.plot(hist.history[metric],label='train')
            val_key='val_'+metric
            if val_key in hist.history: plt.plot(hist.history[val_key],label='validation')
            plt.xlabel('Epoch'); plt.ylabel(metric.replace('_',' ').title()); plt.legend(); plt.tight_layout()
            plt.savefig(out/'plots'/f'training_{metric}.png',dpi=160); plt.close()
    cm=confusion_matrix(yte,(prob>=0.5).astype(int),labels=[0,1])
    fig,ax=plt.subplots(); ConfusionMatrixDisplay(cm,display_labels=['Normal','Attack']).plot(ax=ax,values_format='d',colorbar=False)
    fig.tight_layout(); fig.savefig(out/'plots/confusion_matrix.png',dpi=160); plt.close(fig)
    plt.figure(); plt.hist(prob[yte==0],bins=30,alpha=.7,label='Normal'); plt.hist(prob[yte==1],bins=30,alpha=.7,label='Attack')
    plt.xlabel('Predicted attack probability'); plt.ylabel('Count'); plt.legend(); plt.tight_layout()
    plt.savefig(out/'plots/prediction_probability_histogram.png',dpi=160); plt.close()
    # Mean attention over TRAIN attack samples only: never use zero-day test rows to rank features.
    attack_idx=np.flatnonzero(ytr==1)
    att=[]
    for start in range(0,len(attack_idx),a.batch_size):
        att.append(attention_model.predict(Xtr[attack_idx[start:start+a.batch_size]],verbose=0))
    mean_att=np.concatenate(att,axis=0).mean(axis=(0,2))
    order=np.argsort(mean_att)[::-1]
    fas=[{'feature':features[int(i)],'mean_attention':float(mean_att[i]),'rank':rank+1} for rank,i in enumerate(order)]
    (out/'feature_attention_scores.json').write_text(json.dumps(fas,indent=2))
    plt.figure(figsize=(12,5)); plt.plot(np.arange(1,len(order)+1),mean_att[order],marker='.')
    plt.xticks(np.arange(1,len(order)+1),[features[i] for i in order],rotation=90)
    plt.xlabel('Feature (sorted by mean attention)'); plt.ylabel('Mean attention score'); plt.tight_layout()
    plt.savefig(out/'plots/mean_feature_attention_scores.png',dpi=160); plt.close()
    # Transparent heuristic: largest absolute adjacent drop in sorted FAS; top-15 comparison also reported.
    sorted_scores=mean_att[order]
    if len(sorted_scores)>1:
        drops=sorted_scores[:-1]-sorted_scores[1:]
        cut=int(np.argmax(drops)+1)
    else: cut=1
    selection={'method':'largest adjacent drop in descending mean attention (heuristic; paper threshold underspecified)',
               'selected_count':cut,'selected_features':[features[int(i)] for i in order[:cut]],
               'top_15_comparison':[features[int(i)] for i in order[:min(15,len(order))]],
               'note':'This heuristic is an implementation choice, not a claim that the paper uniquely specifies this threshold.'}
    (out/'feature_selection.json').write_text(json.dumps(selection,indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('classification_report','feature_names')},indent=2))
    print('Saved model, metrics, FAS, feature selection, and plots to',out)
if __name__=='__main__': main()

"""Standalone top-feature and training-only class-weight experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

BASE=Path('data/processed/model_ready'); RANK=Path('reports/feature_importance_ranking.csv'); OUT=Path('reports/reduced_balanced_experiment.json')
def load(path, cols):
    use=['Patient_ID','ICULOS','SepsisLabel',*cols]; dt={'Patient_ID':'string','ICULOS':'float32','SepsisLabel':'int8',**{c:'float32' for c in cols}}
    return pd.read_csv(path,usecols=use,dtype=dt).sort_values(['Patient_ID','ICULOS'],kind='mergesort').reset_index(drop=True)
def score(frame,p,t):
    rows=[]
    for pid,g in frame.assign(p=p).groupby('Patient_ID',sort=False):
        g=g.sort_values('ICULOS'); run=0; hit=0
        for v in g.p:
            run=run+1 if v>=t else 0
            if run>=2: hit=1; break
        rows.append((int(g.SepsisLabel.max()),hit,float(g.p.max())))
    d=pd.DataFrame(rows,columns=['y','pred','score']); tn,fp,fn,tp=confusion_matrix(d.y,d.pred,labels=[0,1]).ravel(); pr=precision_score(d.y,d.pred,zero_division=0); re=recall_score(d.y,d.pred,zero_division=0)
    return {'accuracy':float(accuracy_score(d.y,d.pred)),'precision':float(pr),'recall':float(re),'f1':float(f1_score(d.y,d.pred,zero_division=0)),'specificity':float(tn/max(tn+fp,1)),'roc_auc':float(roc_auc_score(d.y,d.score)),'pr_auc':float(average_precision_score(d.y,d.score)),'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn)}
def main():
    rank=pd.read_csv(RANK); name_col=next(c for c in ['feature','Feature','feature_name'] if c in rank.columns); cols=[c for c in rank[name_col].tolist() if c not in {'Patient_ID','ICULOS','SepsisLabel'}][:50]
    train=load(BASE/'train.csv',cols); val=load(BASE/'validation.csv',cols)
    model=XGBClassifier(n_estimators=220,max_depth=4,learning_rate=.05,min_child_weight=5,subsample=.85,colsample_bytree=.85,reg_lambda=6,gamma=.05,scale_pos_weight=27.271881390593048,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',n_jobs=2,random_state=42)
    model.fit(train[cols],train.SepsisLabel,eval_set=[(val[cols],val.SepsisLabel)],verbose=False); prob=model.predict_proba(val[cols])[:,1]
    best=None
    for t in np.arange(.30,.91,.02):
        m=score(val,prob,float(t)); cand=(m['f1'],m['precision'],float(t),m)
        if best is None or cand[:2]>best[:2]: best=cand
    out={'feature_count':len(cols),'features':cols,'threshold':best[2],'validation_metrics':best[3],'training_balanced_by_class_weight':True,'test_loaded':False,'production_modified':False}; OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

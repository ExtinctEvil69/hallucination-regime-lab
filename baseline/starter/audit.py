"""Evaluate locked models on fresh dossiers under prespecified distribution shifts."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from evaluate import labels, simulated
from bayesian import filter_batch

def generate(seed, scenario, count=500):
    rng = np.random.default_rng(seed)
    transition = np.array([[.85,.13,.02],[.12,.65,.23],[.06,.10,.84]])
    if scenario == 'independent':
        stationary = np.linalg.matrix_power(transition,100)[0]
        transition[:] = stationary
    z = rng.choice(3,count,p=[.84,.14,.02])
    features, truth = [], []
    for turn in range(10):
        apparent = z.copy()
        mask = rng.random(count)<.5
        if scenario == 'confident_errors':
            apparent[(z==2)&mask] = 0
        x = rng.normal(np.array([-.6,.8,.6])+apparent[:,None]*[-.44,.47,-.13],[.8,.8,.35])
        features.append(x);truth.append(z==2)
        z=(rng.random((count,1))>transition[z].cumsum(1)[:,:-1]).sum(1)
    return np.stack(features,1),np.stack(truth,1)

def risk_set(y):
    return (y.cumsum(1)-y)==0

def features(x):
    previous=np.zeros_like(x);previous[:,1:]=x[:,:-1]
    turn=np.broadcast_to(np.arange(x.shape[1]),x.shape[:2])
    return np.concatenate([x,previous,turn[...,None]/10],axis=-1)

def main():
    folder=Path(__file__).resolve().parent/'research_results'
    with np.load(folder/'posterior.npz') as p:
        draws=[p[key].reshape((-1,)+p[key].shape[2:])[::10] for key in ['pi','A','weights','mean','var']]
        center,scale,k=p['center'],p['scale'],p['mapping'][2]
    docs=simulated(180);train=np.random.default_rng(81).permutation(180)[:108]
    x=(np.array([[t['x'] for t in docs[i]['turns']] for i in train])-center)/scale
    y=np.array([labels(docs[i])==2 for i in train]);eligible=risk_set(y)
    clf=LogisticRegression().fit(features(x)[eligible],y[eligible])
    threshold=json.loads((folder/'report.json').read_text())['threshold']
    rows=[]
    for scenario in ['original','independent','confident_errors']:
        x,y=generate(48933,scenario); x=(x-center)/scale; eligible=risk_set(y)
        f=np.mean([filter_batch(x,*d)[...,k] for d in zip(*draws)],axis=0)
        v=features(x);l=clf.predict_proba(v.reshape(-1,v.shape[-1]))[:,1].reshape(y.shape)
        for name,s in [('Bayesian HMM',f),('Supervised logistic',l)]:
            yy=y[eligible];ss=s[eligible]
            rows.append(dict(scenario=scenario,model=name,dossiers=len(y),onsets=int(yy.sum()),auc=roc_auc_score(yy,ss),ap=average_precision_score(yy,ss),brier=brier_score_loss(yy,ss)))
        counts=Counter()
        for target,score in zip(y,f):
            event=np.flatnonzero(target);alarm=np.flatnonzero(score>=threshold)
            if not len(event): outcome='false stop' if len(alarm) else 'clean'
            elif not len(alarm):outcome='miss'
            else:outcome=['early','on time','late'][1+int(np.sign(alarm[0]-event[0]))]
            counts[outcome]+=1
        print(scenario,dict(counts))
    df=pd.DataFrame(rows);df.to_csv(folder/'stress_test.csv',index=False);print(df.to_string(index=False))
if __name__=='__main__':main()

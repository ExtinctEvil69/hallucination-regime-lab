import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from hmm import HMM, lse

NAMES = ['Grounded','Uncertain','Hallucinating']

def auc(y,s):
    """Average ranks for ties. Single-class risk sets have undefined AUROC."""
    y,s = np.asarray(y,bool),np.asarray(s)
    p = y.sum()
    if p == 0 or p == len(y):
        return None
    _,inv,counts = np.unique(s,return_inverse=True,return_counts=True)
    ranks = (counts.cumsum()-(counts-1)/2)[inv]
    return float((ranks[y].sum()-p*(p+1)/2)/(p*(len(y)-p)))

def load_guard(path):
    with np.load(path) as a:
        h = HMM.__new__(HMM)
        for key in ('pi','A','mean','var','weights'):
            setattr(h,key,a[key])
        return h,a['center'],a['scale'],int(a['k']),float(a['threshold'])

def simulated(n):
    rng = np.random.default_rng(2026)
    transition = np.array([[.85,.13,.02],[.12,.65,.23],[.06,.10,.84]])
    result = []
    for i in range(n):
        state = rng.choice(3,p=[.84,.14,.02])
        rows = []
        for _ in range(10):
            x = rng.normal(np.array([-.6,.8,.6])+state*np.array([-.44,.47,-.13]),[.8,.8,.35])
            rows.append(dict(x=x,label=NAMES[state]))
            state = rng.choice(3,p=transition[state])
        result.append(dict(id=i,turns=rows))
    return result

def labels(doc):
    try:
        return np.array([NAMES.index(t['label']) for t in doc['turns']])
    except (ValueError,KeyError):
        raise ValueError(f"Review ungraded turns in trajectory {doc['id']} before evaluation.") from None

def riskset(y):
    first = np.flatnonzero(y==2)
    return slice(0,int(first[0])+1 if len(first) else len(y))

def run(docs,out,synthetic):
    if len(docs)<10 or len(set(d['id'] for d in docs)) != len(docs):
        raise ValueError('Use at least 10 distinct trajectory IDs.')
    if any(d.get('guarded') for d in docs):
        raise ValueError('Evaluation requires uninterrupted trajectories.')
    if len(set(d.get('model') for d in docs))!=1:
        raise ValueError('Use one generator per experiment.')
    out.mkdir(parents=True,exist_ok=True)
    order = np.random.default_rng(81).permutation(len(docs))
    train,val,test = np.split(order,[int(.6*len(order)),int(.8*len(order))])
    x = [np.asarray([t['x'] for t in d['turns']],float) for d in docs]
    if any(a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() for a in x):
        raise ValueError('Each turn needs three finite features.')
    flat = np.concatenate([x[i] for i in train])
    center,scale = flat.mean(0),flat.std(0).clip(1e-6)
    x = [(a-center)/scale for a in x]
    seq = [x[i] for i in train]
    best = -np.inf
    for seed in (17,39,71):
        model = HMM(seq,seed)
        history = model.fit(seq)
        score = sum(lse(model.forward(model.emissions(a)[1])[-1],0) for a in seq)
        if score>best:
            h,trace,best = model,history,score
    prior = np.concatenate([np.exp((f:=h.forward(h.emissions(a)[1]))-lse(f,1)[:,None]) for a in seq]).mean(0)
    def posterior(i):
        e = h.emissions(x[i])[1]
        f = h.forward(e)
        return np.exp(f-lse(f,1)[:,None])
    vy = np.concatenate([labels(docs[i]) for i in val])
    if len(set(vy))<3:
        raise ValueError('Validation needs examples of all three semantic states. Inspect labels and expand the dataset.')
    vp = np.concatenate([posterior(i) for i in val])
    # mapping[label] gives its latent state index; weight classes equally.
    mapping = max(itertools.permutations(range(3)),key=lambda m: sum(vp[vy==j,m[j]].mean() for j in range(3)))
    k = mapping[2]
    def metrics_input(i):
        y = labels(docs[i])
        e = h.emissions(x[i])[1] + np.log(prior)
        static = np.exp(e-lse(e,1)[:,None])[:,k]
        scores = np.column_stack([posterior(i)[:,k],static,-x[i][:,0],x[i][:,1],-x[i][:,2]])
        return y,scores
    def eligible(i):
        y,s = metrics_input(i)
        return np.column_stack([y==2,s])[riskset(y)]
    v = np.concatenate([eligible(i) for i in val])
    if len(np.unique(v[:,0]))<2:
        raise ValueError('Both onset classes required in validation.')
    chosen = max(range(3,6),key=lambda j:auc(v[:,0],v[:,j]))
    neg = v[v[:,0]==0,1]
    threshold = float(np.nextafter(np.quantile(neg,.95,method='higher'),np.inf))
    report = dict(data='synthetic' if synthetic else 'MLX',train=len(train),validation=len(val),test=len(test),
                  train_loglikelihood_trace=trace,states=dict(zip(NAMES,mapping)),transition=h.A.tolist(),
                  static_feature=['negative_logp','entropy','negative_cosine'][chosen-3],alert_threshold=threshold)
    rows = [eligible(i) for i in test]
    pooled = np.concatenate(rows)
    report['test_first_onsets'] = int(pooled[:,0].sum())
    report['test_at_risk_turns'] = len(pooled)
    def summary(a):
        return [auc(a[:,0],a[:,j]) for j in [1,2,chosen]]
    report['onset_auc'] = dict(zip(['filter','emissions_only','single_feature'],summary(pooled)))
    report['onset_sensitivity'] = float((pooled[pooled[:,0]==1,1]>=threshold).mean()) if pooled[:,0].any() else None
    report['at_risk_false_alarm_rate'] = float((pooled[pooled[:,0]==0,1]>=threshold).mean()) if (pooled[:,0]==0).any() else None
    rng = np.random.default_rng(99)
    delta = []
    for _ in range(500):
        a,b,c = summary(np.concatenate([rows[j] for j in rng.integers(len(rows),size=len(rows))]))
        if a is not None:
            delta.append([a-b,a-c])
    report['valid_bootstrap_replicates'] = len(delta)
    report['delta_95pct_intervals'] = dict(zip(['filter_minus_emissions','filter_minus_feature'],np.quantile(delta,[.025,.975],axis=0).T.tolist())) if delta else {}
    np.savez(out/'guard.npz',**vars(h),center=center,scale=scale,k=k,threshold=threshold)
    with (out/'test_audit.csv').open('w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['trajectory','turn','label','online_risk','static_gmm','feature_score','viterbi','at_risk','alert'])
        for i in test:
            y,s = metrics_input(i)
            path = h.decode(h.emissions(x[i])[1])
            for t,state in enumerate(path):
                writer.writerow([docs[i]['id'],t+1,NAMES[y[t]],*s[t,[0,1,chosen-1]],NAMES[mapping.index(state)],t<riskset(y).stop,s[t,0]>=threshold])
    (out/'splits.json').write_text(json.dumps({name:[docs[i]['id'] for i in idx] for name,idx in [('train',train),('validation',val),('test',test)]}))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='train_loglikelihood_trace'},indent=2))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo',action='store_true')
    group.add_argument('--dir',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    docs = simulated(180) if args.demo else [json.loads(p.read_text()) for p in sorted(args.dir.glob('*.json'))]
    run(docs,args.out,args.demo)

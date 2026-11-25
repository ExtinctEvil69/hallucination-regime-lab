"""Reproducible synthetic Bayesian experiment; no real-model claims."""
import argparse,json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from scipy.special import logsumexp
from evaluate import simulated,labels,NAMES,riskset,auc
from hmm import HMM
from bayesian import gibbs,posterior_predictions,filter_batch

def main():
    p=argparse.ArgumentParser();p.add_argument('--draws',type=int,default=400);p.add_argument('--warmup',type=int,default=200);p.add_argument('--out',default='research_results');a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=True)
    docs=simulated(180); X=np.array([[t['x'] for t in d['turns']] for d in docs]); Y=np.array([labels(d) for d in docs])
    tr,va,te=np.split(np.random.default_rng(81).permutation(len(X)),[108,144])
    center=X[tr].reshape(-1,3).mean(0);scale=X[tr].reshape(-1,3).std(0).clip(1e-6);X=(X-center)/scale
    models=[]
    for seed in [17,39,71]:
        h=HMM(X[tr],seed);h.fit(X[tr]);models.append(h)
    h=max(models,key=lambda h:sum(logsumexp(h.forward(h.emissions(x)[1])[-1]) for x in X[tr]))
    params=(h.pi,h.A,h.weights,h.mean,h.var)
    vp=filter_batch(X[va],*params);mapping=max(itertools.permutations(range(3)),key=lambda m:sum(vp[...,m[j]][Y[va]==j].mean() for j in range(3)));k=mapping[2]
    chains=[gibbs(X[tr],h,a.draws,a.warmup,seed) for seed in range(4)]
    posterior=tuple(np.concatenate(v) for v in zip(*chains))
    np.savez_compressed(out/'posterior.npz',**{name:np.stack(v) for name,v in zip(['pi','A','weights','mean','var'],zip(*chains))},mapping=mapping,center=center,scale=scale)
    risk,forecast,attr=posterior_predictions(X[te],posterior,k)
    valrisk=posterior_predictions(X[va],posterior,k)[0].mean(0)
    clean=valrisk[(Y[va]!=2).all(1)].max(1)
    threshold=float(np.nextafter(np.sort(clean)[max(0,len(clean)-int(.05*len(clean))-1)],np.inf)) if len(clean) else None
    prior=filter_batch(X[tr],*params).mean((0,1))
    static=filter_batch(X[te],prior,np.tile(prior,(3,1)),*params[2:])[...,k]
    em=filter_batch(X[te],*params)[...,k]
    mean=risk.mean(0); lo,hi=np.quantile(risk,[.025,.975],axis=0)
    rows=[]
    for i,idx in enumerate(te):
        for t in range(X.shape[1]):
            rows.append(dict(dossier=int(idx),turn=t+1,label=NAMES[Y[idx,t]],target=int(Y[idx,t]==2),at_risk=t<riskset(Y[idx]).stop,bayes=mean[i,t],lower=lo[i,t],upper=hi[i,t],em=em[i,t],static=static[i,t],next_risk=forecast.mean(0)[i,t],logp_contribution=attr.mean(0)[i,t,0],entropy_contribution=attr.mean(0)[i,t,1],cosine_contribution=attr.mean(0)[i,t,2]))
    df=pd.DataFrame(rows);df.to_csv(out/'predictions.csv',index=False)
    d=df[df.at_risk]; metrics={m:{'auc':roc_auc_score(d.target,d[m]),'ap':average_precision_score(d.target,d[m]),'brier':brier_score_loss(d.target,d[m])} for m in ['bayes','em','static']}
    A=posterior[1][:,mapping,:][:,:,mapping];np.save(out/'transition.npy',A)
    rng=np.random.default_rng(99);groups=[g for _,g in d.groupby('dossier')];delta=[]
    for _ in range(500):
        b=pd.concat([groups[j] for j in rng.integers(len(groups),size=len(groups))]);u=auc(b.target,b.bayes);v=auc(b.target,b.static)
        if u is not None:delta.append(u-v)
    report=dict(data='synthetic',draws_per_chain=a.draws,warmup=a.warmup,chains=4,metrics=metrics,test_dossiers=len(te),onsets=int(d.target.sum()),at_risk=len(d),threshold=threshold,clean_validation=len(clean),bayes_minus_static_interval=np.quantile(delta,[.025,.975]).tolist(),limitations='Frozen training posterior ensemble; latent-state alignment approximate; no real LLM or causal factor effects measured.')
    try:
        import arviz as az
        idata=az.from_dict(posterior={'A':np.stack([c[1][:,mapping,:][:,:,mapping] for c in chains])})
        report['transition_rhat_max']=float(az.rhat(idata).A.max());report['transition_ess_min']=float(az.ess(idata).A.min())
    except ImportError:report['diagnostics']='ArviZ unavailable'
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

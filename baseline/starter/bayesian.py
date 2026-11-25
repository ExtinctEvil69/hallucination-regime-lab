"""Conjugate GMM-HMM Gibbs sampling, vectorized across equal-length dossiers."""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from hmm import HMM

def categorical(p,rng):
    return (rng.random(p.shape[:-1])[...,None] > np.cumsum(p,axis=-1)[...,:-1]).sum(-1)

def emissions(x,w,mu,v,mask=None):
    mask=np.ones(x.shape[-1],bool) if mask is None else mask
    return np.log(w)-.5*(np.log(2*np.pi*v[...,mask])+(x[...,None,None,mask]-mu[...,mask])**2/v[...,mask]).sum(-1)

def filter_batch(x,pi,A,w,mu,v,mask=None):
    e=logsumexp(emissions(x,w,mu,v,mask),axis=-1)
    f=np.empty_like(e)
    f[:,0]=np.log(pi)+e[:,0]
    f[:,0]-=logsumexp(f[:,0],axis=-1,keepdims=True)
    for t in range(1,x.shape[1]):
        f[:,t]=e[:,t]+logsumexp(f[:,t-1,:,None]+np.log(A),axis=-2)
        f[:,t]-=logsumexp(f[:,t],axis=-1,keepdims=True)
    return np.exp(f)

def gibbs(x,initial,draws=400,warmup=200,seed=1):
    rng=np.random.default_rng(seed)
    pi,A,w,mu,v=[a.copy() for a in (initial.pi,initial.A,initial.weights,initial.mean,initial.var)]
    n,T,d=x.shape
    reference=(initial.weights[...,None]*initial.mean).sum(1)
    saved=[]
    for it in range(warmup+draws):
        f=filter_batch(x,pi,A,w,mu,v)
        z=np.empty((n,T),int); z[:,-1]=categorical(f[:,-1],rng)
        for t in range(T-2,-1,-1):
            p=f[:,t]*A[:,z[:,t+1]].T
            z[:,t]=categorical(p/p.sum(-1,keepdims=True),rng)
        comp=emissions(x,w,mu,v)[np.arange(n)[:,None],np.arange(T)[None,:],z]
        m=categorical(np.exp(comp-logsumexp(comp,axis=-1,keepdims=True)),rng)
        pi=rng.dirichlet(1+np.bincount(z[:,0],minlength=3))
        counts=np.bincount((3*z[:,:-1]+z[:,1:]).ravel(),minlength=9).reshape(3,3)
        A=np.stack([rng.dirichlet(1+row) for row in counts])
        for j in range(3):
            w[j]=rng.dirichlet(1+np.bincount(m[z==j],minlength=2))
            for k in range(2):
                a=x[(z==j)&(m==k)]; N=len(a); kn=.5+N
                mn=a.sum(0)/kn
                bn=2+.5*((a-mn)**2).sum(0)+.25*mn**2
                v[j,k]=1/rng.gamma(3+N/2,1/bn,size=d)
                mu[j,k]=rng.normal(mn,np.sqrt(v[j,k]/kn))
        if it>=warmup:
            center=(w[...,None]*mu).sum(1)
            _,idx=linear_sum_assignment(((reference[:,None]-center[None,:])**2).sum(-1))
            saved.append((pi[idx],A[np.ix_(idx,idx)],w[idx].copy(),mu[idx].copy(),v[idx].copy()))
    return tuple(np.stack(a) for a in zip(*saved))

def posterior_predictions(x,posterior,k):
    risks=[]; forecasts=[]; attribution=[]
    for pi,A,w,mu,v in zip(*posterior):
        f=filter_batch(x,pi,A,w,mu,v)
        risks.append(f[...,k]); forecasts.append((f@A)[...,k])
        attribution.append(np.stack([f[...,k]-filter_batch(x,pi,A,w,mu,v,np.arange(x.shape[-1])!=d)[...,k] for d in range(x.shape[-1])],axis=-1))
    return np.asarray(risks),np.asarray(forecasts),np.asarray(attribution)

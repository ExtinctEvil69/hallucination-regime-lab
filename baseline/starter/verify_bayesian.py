import numpy as np
from hmm import HMM
from bayesian import filter_batch,gibbs,posterior_predictions
rng=np.random.default_rng(9);x=rng.normal(size=(8,5,3));h=HMM(x)
p=(h.pi,h.A,h.weights,h.mean,h.var)
f=filter_batch(x,*p)
for i in range(len(x)):
    prev=None
    for t in range(x.shape[1]):
        prev=h.step(x[i,t],prev);np.testing.assert_allclose(f[i,t],prev,atol=1e-12)
np.testing.assert_allclose(filter_batch(x[:,:3],*p),f[:,:3])
draw=gibbs(x,h,10,5)
np.testing.assert_allclose(draw[0].sum(-1),1)
np.testing.assert_allclose(draw[1].sum(-1),1)
assert (draw[4]>0).all()
r,q,a=posterior_predictions(x,draw,2)
assert r.shape==(10,8,5) and a.shape==(10,8,5,3)
assert np.isfinite(r).all() and ((q>=0)&(q<=1)).all()
print("Passed Bayesian filter parity, prefix invariance, Gibbs probability and forecast checks.")

"""Check exact inference against exhaustive enumeration on a small chain."""
from itertools import product
import numpy as np
from hmm import HMM
from evaluate import auc
from collect import dossier,grade

rng = np.random.default_rng(3)
x = rng.normal(size=(5,3))
h = HMM([x])
_,e = h.emissions(x)
paths = list(product(range(3),repeat=len(x)))
def joint(path):
    return (np.log(h.pi[path[0]])+sum(e[t,j] for t,j in enumerate(path))
            +sum(np.log(h.A[i,j]) for i,j in zip(path,path[1:])))
scores = np.array([joint(p) for p in paths])
ll,start,edges,r = h.expectations(x)
np.testing.assert_allclose(ll,np.logaddexp.reduce(scores))
np.testing.assert_array_equal(h.decode(e),paths[scores.argmax()])
p = np.exp(scores-ll)
states = np.eye(3)[np.array(paths)]
np.testing.assert_allclose(r.sum(2),np.einsum('n,ntk->tk',p,states))
np.testing.assert_allclose(edges,np.einsum('n,nti,ntj->ij',p,states[:,:-1],states[:,1:]))
previous = None
for t,row in enumerate(x):
    previous = h.step(row,previous)
    prefix = list(product(range(3),repeat=t+1))
    mass = np.bincount([s[-1] for s in prefix],weights=[np.exp(joint(s)) for s in prefix])
    np.testing.assert_allclose(previous,mass/mass.sum())
trace = h.fit([x,x[::-1]])
assert min(np.diff(trace)) >= -1e-6
assert auc([0,1],[1,1])==.5
assert grade('unknown',None)=='Grounded' and grade('RED-03','RED-04')=='Hallucinating'
assert dossier(5)==dossier(5)
print('Passed enumeration, online prefix, EM, tie scoring, and fixture checks.')

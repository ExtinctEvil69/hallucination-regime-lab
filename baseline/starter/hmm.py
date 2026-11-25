"""Log-domain diagonal GMM-HMM. A row indexes the previous state."""
import numpy as np

def lse(x, axis):
    return np.logaddexp.reduce(x, axis=axis)

class HMM:
    def __init__(self, sequences, seed=7, states=3, mixtures=2):
        rng = np.random.default_rng(seed)
        x = np.concatenate(sequences)
        self.pi = np.full(states, 1 / states)
        self.A = .5 * np.eye(states) + .5 / states
        self.weights = np.full((states, mixtures), 1 / mixtures)
        self.mean = x[rng.choice(len(x), (states, mixtures))].copy()
        self.var = np.broadcast_to(x.var(0) + .1, self.mean.shape).copy()

    def emissions(self, x):
        residual = x[:, None, None, :] - self.mean
        comp = np.log(self.weights) - .5 * (
            np.log(2*np.pi*self.var) + residual**2 / self.var).sum(-1)
        return comp, lse(comp, 2)

    def forward(self, emission):
        f = np.empty_like(emission)
        f[0] = np.log(self.pi) + emission[0]
        transition = np.log(self.A)
        for t in range(1, len(f)):
            f[t] = emission[t] + lse(f[t-1, :, None] + transition, 0)
        return f

    def step(self, x, previous=None):
        """One causal update. previous is the previous normalized probability vector."""
        prior = self.pi if previous is None else previous @ self.A
        score = np.log(prior) + self.emissions(np.asarray(x)[None])[1][0]
        return np.exp(score - lse(score, 0))

    def decode(self, emission):
        score = np.log(self.pi) + emission[0]
        back = []
        for row in emission[1:]:
            pairs = score[:, None] + np.log(self.A)
            back.append(pairs.argmax(0))
            score = row + pairs.max(0)
        path = [score.argmax()]
        for pointers in reversed(back):
            path.append(pointers[path[-1]])
        return path[::-1]

    def expectations(self, x):
        comp, emission = self.emissions(x)
        f = self.forward(emission)
        ll = lse(f[-1], 0)
        b = np.zeros_like(f)
        transition = np.log(self.A)
        for t in range(len(b)-2, -1, -1):
            b[t] = lse(transition + emission[t+1] + b[t+1], 1)
        gamma = np.exp(f + b - ll)
        xi = np.exp(f[:-1, :, None] + transition + (emission[1:] + b[1:])[:, None, :] - ll)
        resp = gamma[:, :, None] * np.exp(comp - emission[:, :, None])
        return ll, gamma[0], xi.sum(0), resp

    def fit(self, sequences, iterations=45):
        history = []
        for _ in range(iterations):
            start = np.full_like(self.pi, 1e-8)
            pairs = np.full_like(self.A, 1e-8)
            mass = np.zeros_like(self.weights)
            moment = np.zeros_like(self.mean)
            square = np.zeros_like(self.var)
            total = 0.
            for x in sequences:
                ll, initial, counts, r = self.expectations(x)
                total += ll
                start += initial
                pairs += counts
                mass += r.sum(0)
                moment += np.einsum('tkm,td->kmd', r, x)
                square += np.einsum('tkm,td->kmd', r, x*x)
            history.append(float(total))
            if len(history)>1 and abs(total-history[-2]) < 1e-5*(1+abs(total)):
                break
            self.pi = start/start.sum()
            self.A = pairs/pairs.sum(1, keepdims=True)
            w = mass + 1e-8
            self.weights = w/w.sum(1, keepdims=True)
            active = mass > 1e-8
            self.mean[active] = moment[active]/mass[active,None]
            self.var[active] = np.maximum(square[active]/mass[active,None]-self.mean[active]**2, .01)
        return history

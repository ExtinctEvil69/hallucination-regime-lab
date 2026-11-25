"""Exact numerical oracles: brute-force enumeration over all 3^T paths."""
import itertools

import numpy as np
import pytest

from regime_lab.models.bayes import BayesHMM, _perm_theta, cov_logA, ffbs
from regime_lab.models.emissions import GMMEmissions
from regime_lab.models.hmm import EMHMM, filter_probs, forward, map_states, smooth, viterbi

K, M, D, T = 3, 2, 3, 5


def random_model(seed=0):
    rng = np.random.default_rng(seed)
    em = GMMEmissions(rng.dirichlet(np.ones(M), K), rng.normal(0, 1.5, (K, M, D)),
                      rng.uniform(.3, 2, (K, M, D)))
    return rng, em, rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K) * 2, K)


def enumerate_paths(log_b, log_pi, logA):
    """Joint log p(z, x) for every path of one sequence; logA (T,K,K) or (K,K)."""
    T_, K_ = log_b.shape
    out = {}
    for z in itertools.product(range(K_), repeat=T_):
        lp = log_pi[z[0]] + log_b[0, z[0]]
        for t in range(1, T_):
            At = logA if logA.ndim == 2 else logA[t]
            lp += At[z[t - 1], z[t]] + log_b[t, z[t]]
        out[z] = lp
    return out


@pytest.mark.parametrize("covariate", [False, True])
def test_forward_backward_viterbi_match_enumeration(covariate):
    rng, em, pi, A = random_model(1)
    X = rng.normal(size=(2, T, D))
    mask = np.ones((2, T), bool)
    mask[1, 2] = False                      # an ineligible turn is marginalised
    lb = em.log_b(X, mask)
    if covariate:
        theta = rng.normal(0, 1, (K, K, 3)); theta[:, -1] = 0
        logA = cov_logA(theta, rng.normal(size=(2, T, 2)))
    else:
        logA = np.log(A)
    gamma, xi, ll = smooth(lb, np.log(pi), logA)
    path, _ = viterbi(lb, np.log(pi), logA)
    f, q, _, ll2 = filter_probs(lb, np.log(pi), logA)
    for n in range(2):
        lA = logA if logA.ndim == 2 else logA[n]
        paths = enumerate_paths(lb[n], np.log(pi), lA)
        vals = np.array(list(paths.values())); keys = np.array(list(paths.keys()))
        L = np.logaddexp.reduce(vals)
        assert np.isclose(L, ll[n], atol=1e-10) and np.isclose(L, ll2[n], atol=1e-10)
        post = np.exp(vals - L)
        for t in range(T):
            marg = np.array([post[keys[:, t] == j].sum() for j in range(K)])
            assert np.allclose(marg, gamma[n, t], atol=1e-10)
        for t in range(T - 1):
            pair = np.array([[post[(keys[:, t] == i) & (keys[:, t + 1] == j)].sum() for j in range(K)]
                             for i in range(K)])
            assert np.allclose(pair, xi[n, t], atol=1e-10)
        assert tuple(path[n]) == tuple(keys[np.argmax(vals)])
        # causal filter equals enumeration over prefixes only
        for t in range(T):
            pre = enumerate_paths(lb[n, :t + 1], np.log(pi), lA if lA.ndim == 2 else lA[:t + 1])
            pv = np.array(list(pre.values())); pk = np.array(list(pre.keys()))
            pp = np.exp(pv - np.logaddexp.reduce(pv))
            assert np.allclose([pp[pk[:, -1] == j].sum() for j in range(K)], f[n, t], atol=1e-10)


def test_prefix_invariance_and_no_cross_boundary():
    rng, em, pi, A = random_model(2)
    X = rng.normal(size=(3, 6, D)); mask = np.ones((3, 6), bool)
    f_full, *_ = filter_probs(em.log_b(X, mask), np.log(pi), np.log(A))
    f_pre, *_ = filter_probs(em.log_b(X[:, :4], mask[:, :4]), np.log(pi), np.log(A))
    assert np.allclose(f_full[:, :4], f_pre)          # future turns never change past filter
    f_one, *_ = filter_probs(em.log_b(X[1:2], mask[1:2]), np.log(pi), np.log(A))
    assert np.allclose(f_one[0], f_full[1])            # rows (dossiers) are independent


def test_single_turn_has_no_transition():
    rng, em, pi, A = random_model(3)
    X = rng.normal(size=(1, 1, D)); mask = np.ones((1, 1), bool)
    gamma, xi, ll = smooth(em.log_b(X, mask), np.log(pi), np.log(A))
    assert xi.shape[1] == 0 and np.isclose(gamma.sum(), 1)


def test_negative_control_equals_static_and_save_load(tmp_path):
    rng = np.random.default_rng(4)
    X = np.concatenate([rng.normal(-1, 1, (40, 6, D)), rng.normal(2, 1, (40, 6, D))])
    mask = np.ones(X.shape[:2], bool)
    m = EMHMM().fit(X, mask, n_starts=2, max_iter=50)
    assert np.allclose(m.A.sum(1), 1) and np.isclose(m.pi.sum(), 1)
    nc = m.negative_control()
    f, _, _ = nc.filter(X, mask)
    assert np.allclose(f, m.static_scores(X, mask), atol=1e-10)
    m.save(tmp_path / "m.npz")
    m2 = EMHMM.load(tmp_path / "m.npz")
    assert np.allclose(m2.filter(X, mask)[0], m.filter(X, mask)[0], atol=1e-8)
    # online step reproduces batch filter
    prev = None
    for t in range(6):
        prev = m.step(X[0, t], prev)
        assert np.allclose(prev, m.filter(X, mask)[0][0, t], atol=1e-10)


def test_em_loglik_nondecreasing_without_floor_effects():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 6, D)) + rng.integers(0, 2, (60, 1, 1)) * 2.5
    m = EMHMM().fit(X, np.ones((60, 6), bool), n_starts=1, max_iter=60, var_floor=1e-6)
    ll = [h["loglik"] for h in m.history if "final" not in h]
    assert np.all(np.diff(ll) > -1e-6 * abs(ll[0]))


def test_marginalising_feature_matches_lower_dim_model():
    rng, em, pi, A = random_model(6)
    X = rng.normal(size=(2, 4, D)); mask = np.ones((2, 4), bool)
    lb_sub = em.log_b(X, mask, dims=[0, 2])
    em2 = GMMEmissions(em.w, em.mu[..., [0, 2]], em.var[..., [0, 2]])
    assert np.allclose(lb_sub, em2.log_b(X[..., [0, 2]], mask))


def test_ffbs_samples_follow_exact_posterior():
    rng, em, pi, A = random_model(7)
    X = rng.normal(size=(1, 3, D)); mask = np.ones((1, 3), bool)
    lb = em.log_b(X, mask)
    paths = enumerate_paths(lb[0], np.log(pi), np.log(A))
    vals = np.array(list(paths.values()))
    exact = np.exp(vals - np.logaddexp.reduce(vals))
    idx = {k: i for i, k in enumerate(paths)}
    counts = np.zeros(len(paths))
    r = np.random.default_rng(0)
    lbr = np.repeat(lb, 20000, axis=0)
    z, _ = ffbs(r, lbr, np.log(pi), np.log(A))
    for row in z:
        counts[idx[tuple(row)]] += 1
    assert np.max(np.abs(counts / counts.sum() - exact)) < 0.01


def test_theta_permutation_preserves_transitions():
    rng = np.random.default_rng(8)
    theta = rng.normal(size=(K, K, 3)); theta[:, -1] = 0
    C = rng.normal(size=(2, 4, 2))
    p = [2, 0, 1]
    A = np.exp(cov_logA(theta, C))
    Ap = np.exp(cov_logA(_perm_theta(theta, p), C))
    assert np.allclose(Ap, A[..., p, :][..., :, p])
    assert np.allclose(_perm_theta(theta, p)[:, -1], 0)


def test_state_mapping_recovers_permutation():
    rng = np.random.default_rng(9)
    labels = rng.choice(["G", "U", "H"], size=(50, 6))
    true = {"G": 1, "U": 2, "H": 0}
    post = np.full((50, 6, 3), .1)
    for (n, t), l in np.ndenumerate(labels):
        post[n, t, true[l]] = .8
    res = map_states(post, labels, np.ones((50, 6), bool))
    assert res["perm"] == [2, 0, 1] and res["stable"]


def test_conjugate_gibbs_recovers_known_transition():
    """Gibbs on well-separated synthetic data recovers A within its credible interval."""
    rng = np.random.default_rng(10)
    A = np.array([[.85, .1, .05], [.2, .7, .1], [.1, .1, .8]])
    N, T_ = 300, 6
    z = np.zeros((N, T_), int); z[:, 0] = rng.integers(0, 3, N)
    for t in range(1, T_):
        z[:, t] = [rng.choice(3, p=A[i]) for i in z[:, t - 1]]
    means = np.array([[-3, 0, 0], [0, 3, 0], [3, -3, 3]], float)
    X = means[z] + rng.normal(0, .5, (N, T_, 3))
    mask = np.ones((N, T_), bool)
    init = EMHMM(M=1).fit(X, mask, n_starts=2, max_iter=100)
    perm = [int(np.argmin(((init.em.state_means()[:, None] - means[None]) ** 2).sum(-1)[:, j]))
            for j in range(3)]
    init = init.permute(perm)
    b = BayesHMM(M=1).fit(X, mask, init=init, chains=2, warmup=150, n_draws=150, seed=1)
    b.align(init.em)
    Ad = b.flat()["A"]
    lo, hi = np.quantile(Ad, .005, axis=0), np.quantile(Ad, .995, axis=0)
    assert np.all((A >= lo - .03) & (A <= hi + .03))

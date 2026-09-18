"""Active/batch selection strategies over group-level candidates in a
target lake's unlabeled pool. All operate on already-deduplicated groups
(gloria.build_tss_groups output) -- never on raw rows -- per the
site/date/rounded-coordinate grouping hard constraint.

Scope: random, farthest-point/Kennard-Stone (+ its query-isolated
entropy-diagnostic variant), unweighted D-optimal, and the proposed weighted
D-optimal. k-medoids and ensemble-uncertainty selectors are deferred.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def farthest_point_ks_order(X_candidates: np.ndarray) -> np.ndarray:
    """Kennard-Stone ordering: start from the two mutually farthest points,
    then repeatedly append the candidate maximizing its minimum distance to
    the already-selected set. Returns a full permutation of candidate
    indices (caller takes the first k for a given budget)."""
    n = len(X_candidates)
    dist = np.linalg.norm(X_candidates[:, None, :] - X_candidates[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [i, j]
    remaining = set(range(n)) - {i, j}
    min_dist_to_selected = np.minimum(dist[i], dist[j])
    while remaining:
        remaining_arr = np.array(sorted(remaining))
        best_local = np.argmax(min_dist_to_selected[remaining_arr])
        best = remaining_arr[best_local]
        selected.append(best)
        remaining.discard(best)
        min_dist_to_selected = np.minimum(min_dist_to_selected, dist[best])
    return np.array(selected)


def shannon_entropy(X: np.ndarray, n_bins: int = 30) -> float:
    """Mean per-band Shannon entropy from histogram density (FSResTL eq. 5),
    diagnostic only -- see selection module docstring."""
    ent = []
    for col in range(X.shape[1]):
        hist, _ = np.histogram(X[:, col], bins=n_bins, density=False)
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        ent.append(-np.sum(p * np.log(p)))
    return float(np.mean(ent))


def entropy_diff_at_k(X_selected: np.ndarray, X_full_pool: np.ndarray) -> float:
    """Delta-H = H(full pool) - H(selected subset) (FSResTL eq. 4). Reported
    as a diagnostic for the ks_entropy_query_isolated baseline -- NOT used to
    choose a stopping k in this project (k is pre-registered), so this never
    touches target labels or influences selection, only describes it."""
    return shannon_entropy(X_full_pool) - shannon_entropy(X_selected)


def support_score(X_target: np.ndarray, X_source: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to the k nearest source-domain neighbors, in whatever
    embedding X_target/X_source already are. Smaller =
    better-supported by the source domain."""
    nn = NearestNeighbors(n_neighbors=min(k, len(X_source))).fit(X_source)
    dist, _ = nn.kneighbors(X_target)
    return dist.mean(axis=1)


def target_density_score(X_target: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to the k nearest neighbors WITHIN the target pool
    itself -- smaller = more representative/typical of the target lake's
    own spectral distribution (rho_i)."""
    k_eff = min(k + 1, len(X_target))  # +1 since a point is its own neighbor
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X_target)
    dist, _ = nn.kneighbors(X_target)
    return dist[:, 1:].mean(axis=1)  # drop self-distance (column 0, always 0)


def greedy_d_optimal_order(Z_candidates: np.ndarray, weights: np.ndarray = None,
                            lam: float = 1e-3, max_k: int = 10) -> np.ndarray:
    """Greedy sequential maximization of log det(lam*I + sum w_i g_i g_i^T),
    g_i = [1, z_i^T]^T. weights=None -> unweighted
    D-optimal (core ablation); weights=(d_source+eps)^-1 * (d_target+eps)^-1
    (inverse mean 10-NN distances, eps=1e-6 in the experiment scripts) ->
    the support- and local-density-weighted design used in the manuscript.
    Per step, the candidate maximizing w_i * g_i^T A^-1 g_i is selected --
    monotone in, hence rank-equivalent to, the one-step weighted
    log-determinant gain. Returns
    the first max_k selections only -- this project's fixed budget never
    exceeds k=10, so building a full n-length ordering would be wasted work
    (O(n^2) for no benefit)."""
    n = len(Z_candidates)
    G = np.column_stack([np.ones(n), Z_candidates])  # [1, z_i^T]
    p = G.shape[1]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    info = lam * np.eye(p)
    remaining = set(range(n))
    order = []
    for _ in range(min(max_k, n)):
        remaining_arr = np.array(sorted(remaining))
        best_gain, best_idx = -np.inf, None
        # log-det update via matrix determinant lemma: avoids recomputing a
        # full pxp determinant from scratch for every candidate at every step
        info_inv = np.linalg.inv(info)
        for idx in remaining_arr:
            g = G[idx]
            gain = w[idx] * (g @ info_inv @ g)  # log(1 + w*g^T A^-1 g), monotone in this term
            if gain > best_gain:
                best_gain, best_idx = gain, idx
        order.append(best_idx)
        remaining.discard(best_idx)
        g = G[best_idx]
        info = info + w[best_idx] * np.outer(g, g)
    return np.array(order)

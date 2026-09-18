"""Group-disjoint support/query sampling for a single target lake.

Candidate unit is the group (site+date-deduplicated), never a raw row, so
near-duplicate technical replicates can never split across support and query.
"""
import numpy as np


def random_support_query_split(group_ids: np.ndarray, k: int,
                                rng: np.random.Generator):
    """Draw k groups as support, the remainder as query. group_ids must
    already be unique (one row per group)."""
    group_ids = np.asarray(group_ids)
    if len(group_ids) <= k:
        raise ValueError(
            f"need more than k={k} groups to leave a nonempty query set, "
            f"got {len(group_ids)} groups"
        )
    perm = rng.permutation(len(group_ids))
    support_idx = perm[:k]
    query_idx = perm[k:]
    return support_idx, query_idx

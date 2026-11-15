"""Record alignment for scores made elsewhere: evaluate() pairs rows by position, so put
every model's rows in the labels' order first."""
import numpy as np


def align_to(reference_ids, ids, *arrays):
    """(reference_ids, ids, *arrays) -> arrays with rows reordered to reference_ids. Duplicate
    ids or a reference id missing from ids raise; extra ids are dropped."""
    ref = list(reference_ids)
    cur = list(ids)
    if len(set(ref)) != len(ref):
        raise ValueError("reference_ids contain duplicates")
    if len(set(cur)) != len(cur):
        raise ValueError("ids contain duplicates")
    pos = {r: i for i, r in enumerate(cur)}
    missing = [r for r in ref if r not in pos]
    if missing:
        raise ValueError(f"{len(missing)} reference ids have no row (e.g. {missing[0]!r})")
    idx = np.array([pos[r] for r in ref], dtype=np.int64)
    out = []
    for a in arrays:
        a = np.asarray(a)
        if a.shape[0] != len(cur):
            raise ValueError(f"an array has {a.shape[0]} rows for {len(cur)} ids")
        out.append(a[idx])
    return out[0] if len(out) == 1 else tuple(out)


def drop_unlabelled(ids, Y, *arrays):
    """Records whose label row has any missing entry (NaN) removed, from `ids`, `Y` and every
    array in `arrays`. evaluate() refuses labels other than 0/1, so a missing label must be
    removed before evaluation, and the same records removed for every model compared on that
    period (align every model to the returned ids). Returns (ids, Y as int, *arrays)."""
    Y = np.asarray(Y, dtype=float)
    keep = ~np.isnan(Y).any(axis=1)
    ids = [i for i, k in zip(ids, keep) if k]
    rest = [np.asarray(a)[keep] for a in arrays]
    return (ids, Y[keep].astype(int), *rest)

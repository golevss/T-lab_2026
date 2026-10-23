import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def graph_features(events, q, t, dim=16, categories=True):
    h = events[events.date < t].copy()
    users = np.sort(h.user_id.unique())
    items = np.sort(h.item_id.unique())
    ui = {v: k for k, v in enumerate(users)}
    ii = {v: k for k, v in enumerate(items)}
    rows = h.user_id.map(ui).tolist()
    cols = h.item_id.map(ii).tolist()
    weights = [1.0] * len(h)
    meta = h.drop_duplicates("item_id")
    if categories:
        offset = len(users)
        for c in ("genre", "content_type"):
            codes, values = pd.factorize(meta[c], sort=True)
            rows.extend((codes + offset).tolist())
            cols.extend(meta.item_id.map(ii).tolist())
            weights.extend([0.5] * len(meta))
            offset += len(values)
    a = coo_matrix(
        (weights, (rows, cols)), shape=(max(rows) + 1, len(items)), dtype="float32"
    ).tocsr()

    a = (
        diags(1 / np.sqrt(np.maximum(np.asarray(a.sum(axis=1)).ravel(), 1e-9)))
        @ a
        @ diags(1 / np.sqrt(np.maximum(np.asarray(a.sum(axis=0)).ravel(), 1e-9)))
    )
    svd = TruncatedSVD(n_components=min(dim, min(a.shape) - 1), n_iter=7, random_state=2026)
    svd.fit(a)
    v = normalize(svd.components_.T).astype("float32")
    allp = np.zeros((len(users), v.shape[1]), dtype="float32")
    recent = allp.copy()
    mass = np.zeros(len(users))
    u = h.user_id.map(ui).to_numpy()
    i = h.item_id.map(ii).to_numpy()
    w = np.exp2(-(t - h.date).dt.days.to_numpy() / 7.0)
    np.add.at(allp, u, v[i])
    np.add.at(recent, u, v[i] * w[:, None])
    np.add.at(mass, u, w)
    allp = normalize(allp)
    recent = normalize(recent)
    out = np.zeros((len(q), 6), dtype="float32")
    for uid, idx in q.groupby("user_id", sort=False).indices.items():
        if uid not in ui:
            out[idx, 5] = 1
            continue
        r = ui[uid]
        cand = v[q.iloc[idx].item_id.map(ii).to_numpy()]
        last = (
            h[h.user_id == uid].sort_values(["date", "item_id"]).tail(3).item_id.map(ii).to_numpy()
        )
        out[idx, 0] = cand @ allp[r]
        out[idx, 1] = cand @ recent[r]
        out[idx, 2] = (cand @ v[last].T).max(axis=1)
        out[idx, 3] = np.log1p(mass[r])
        out[idx, 4] = np.linalg.norm(recent[r] - allp[r])
    frame = pd.DataFrame(
        out,
        columns=[
            "g_cos_all",
            "g_cos_7",
            "g_max_last3",
            "g_recent_mass",
            "g_profile_shift",
            "g_cold",
        ],
    )
    return frame, dict(
        nodes_rows=a.shape[0],
        items=a.shape[1],
        edges=a.nnz,
        dim=v.shape[1],
        categories=categories,
        history_max=str(h.date.max().date()),
        cutoff=str(t.date()),
    )

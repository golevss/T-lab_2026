import numpy as np
import pandas as pd
from features import feature_matrix
from graph_embeddings import graph_features
from scipy.sparse import coo_matrix


def build_features(events, queries, cutoff, item_genres, mode="rich"):
    q = queries.reset_index(drop=True)
    h = events[events.date < cutoff].copy()
    manual = feature_matrix(events, q, cutoff)
    if mode == "manual":
        return manual
    graph, _ = graph_features(events, q, cutoff, 32, True)
    if mode == "graph":
        return pd.concat([manual, graph], axis=1)
    users = np.sort(h.user_id.unique())
    items = np.sort(h.item_id.unique())
    ui = {v: k for k, v in enumerate(users)}
    ii = {v: k for k, v in enumerate(items)}
    u = h.user_id.map(ui).to_numpy()
    i = h.item_id.map(ii).to_numpy()
    a = coo_matrix((np.ones(len(h)), (u, i)), shape=(len(users), len(items))).tocsr()
    c = (a.T @ a).tocsr()
    popularity = c.diagonal().copy()
    c.setdiag(0)
    c.eliminate_zeros()
    for row in range(c.shape[0]):
        lo, hi = c.indptr[row : row + 2]
        values = c.data[lo:hi]
        columns = c.indices[lo:hi]
        c.data[lo:hi] = (
            values / np.sqrt(popularity[row] * popularity[columns]) * values / (values + 10)
        )
        if hi - lo > 100:
            keep = np.argpartition(c.data[lo:hi], -100)[-100:]
            mask = np.ones(hi - lo, dtype=bool)
            mask[keep] = False
            c.data[lo:hi][mask] = 0
    c.eliminate_zeros()
    recent = coo_matrix(
        (np.exp2(-(cutoff - h.date).dt.days.to_numpy() / 7), (u, i)), shape=a.shape
    ).tocsr()
    last = h.sort_values(["date", "item_id"]).groupby("user_id", sort=False).tail(1)
    last_matrix = coo_matrix(
        (np.ones(len(last)), (last.user_id.map(ui), last.item_id.map(ii))), shape=a.shape
    ).tocsr()
    knn = pd.DataFrame(index=q.index)
    known = q.user_id.isin(ui).to_numpy()
    qu = q.loc[known, "user_id"].map(ui).to_numpy()
    qi = q.loc[known, "item_id"].map(ii).to_numpy()
    for name, matrix in [("knn_all", a), ("knn_7", recent), ("knn_last", last_matrix)]:
        scores = matrix @ c
        values = np.zeros(len(q), dtype="float32")
        values[known] = np.asarray(scores[qu, qi]).ravel()
        knn[name] = values
    if mode == "knn":
        return pd.concat([manual, graph, knn], axis=1)
    wide = feature_matrix(events, q, cutoff, windows=(7, 30, 2, 14), halves=(3, 14))
    extra = wide[[c for c in wide if c not in manual]].copy()
    expanded = h[["user_id", "item_id", "date"]].copy()
    expanded["full_genre"] = expanded.item_id.map(item_genres)
    expanded = expanded.explode("full_genre")
    total = h.groupby("user_id").size().to_dict()
    for days in [None, 30]:
        frame = (
            expanded
            if days is None
            else expanded[expanded.date >= cutoff - pd.Timedelta(days=days)]
        )
        counts = frame.groupby(["user_id", "full_genre"]).size().to_dict()
        maximum = []
        mean = []
        for uid, item in zip(q.user_id, q.item_id):
            values = [counts.get((uid, g), 0) for g in item_genres.get(item, ["unknown"])]
            maximum.append(max(values) / (total.get(uid, 0) + 1))
            mean.append(np.mean(values) / (total.get(uid, 0) + 1))
        suffix = "all" if days is None else str(days)
        extra[f"genres_max_{suffix}"] = maximum
        extra[f"genres_mean_{suffix}"] = mean
    return pd.concat([manual, graph, knn, extra], axis=1)


def columns_for(frame, mode):
    graph = [c for c in frame if c.startswith("g_")]
    knn = [c for c in frame if c.startswith("knn_")]
    manual = list(frame.columns[:34])
    if mode == "manual":
        return manual
    if mode == "graph":
        return manual + graph
    if mode == "knn":
        return manual + graph + knn
    return list(frame.columns)

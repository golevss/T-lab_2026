import hashlib
import json
import pathlib
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from features import feature_matrix
from graph import graph_features

ROOT = pathlib.Path(__file__).resolve().parent
COMMIT = "f69775be31fa5779907cf0a92ddedb70037fb5ae"
CONFIGS = {"graph16": (16, True), "graph32": (32, True), "no_categories": (16, False)}


def read(z, name):
    with zipfile.ZipFile(z) as f:
        n = next(
            n
            for n in f.namelist()
            if pathlib.PurePosixPath(n).name == name and not n.startswith("__MACOSX/")
        )
        return pd.read_csv(f.open(n))


def main():
    start = time.perf_counter()
    d = ROOT / "data"
    d.mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    z = d / "data_original.zip"
    if not z.exists():
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{COMMIT}/data_original.zip", z
        )
    raw = read(z, "interactions.csv")
    excluded = np.load(d / "excluded_previous_users.npy")
    rng = np.random.default_rng(2026)
    pool = np.setdiff1d(raw.user_id.unique(), excluded)
    selected = np.sort(rng.choice(pool, 5000, replace=False))
    np.save(d / "selected_users.npy", selected)
    e = raw[raw.user_id.isin(selected)].copy()
    assert set(selected).isdisjoint(excluded)
    expected = e.copy()
    e["date"] = pd.to_datetime(e.last_watch_dt)
    m = read(z, "items.csv")
    m["genre"] = m.genres.fillna("unknown").str.split(",").str[0].str.strip()
    m = m[["item_id", "genre", "content_type", "age_rating"]].copy()
    for c in m.columns[1:]:
        m[c] = m[c].fillna("unknown").astype(str)
    e = e.merge(m, on="item_id", validate="many_to_one")
    e = e.sort_values(["date", "user_id", "item_id"]).reset_index(drop=True)
    assert len(e) == len(expected)
    e.to_pickle(d / "events.pkl")
    qlist = []
    xlist = []
    gs = {k: [] for k in CONFIGS}
    audits = []
    ga = []
    end = e.date.max() + pd.Timedelta(days=1)
    for j in range(8):
        t = end - pd.Timedelta(days=7 * (8 - j))
        h = e[e.date < t]
        future = e[(e.date >= t) & (e.date < t + pd.Timedelta(days=7))]
        catalog = np.sort(h.item_id.unique())
        pop = h.item_id.value_counts().reindex(catalog).to_numpy() ** 0.75
        pop = pop / pop.sum()
        rows = []
        cold = 0
        for uid, g in future.groupby("user_id", sort=True):
            pos = set(g.item_id) & set(catalog)
            cold += len(set(g.item_id) - set(catalog))
            if not pos:
                continue
            neg = set()
            for n, p in ((50, None), (49, pop)):
                target = len(neg) + n
                while len(neg) < target:
                    item = int(rng.choice(catalog, p=p))
                    if item not in pos:
                        neg.add(item)
            rows.extend((j, uid, i, int(i in pos)) for i in sorted(pos | neg))
        q = pd.DataFrame(rows, columns=["snapshot", "user_id", "item_id", "label"]).merge(
            m, on="item_id", validate="many_to_one"
        )
        q["cutoff"] = t
        q["split"] = "train" if j < 6 else ("val" if j == 6 else "test")
        x = feature_matrix(e, q, t)
        qlist.append(q)
        xlist.append(x)
        for name, (dim, cats) in CONFIGS.items():
            ts = time.perf_counter()
            g, info = graph_features(e, q, t, dim, cats)
            gs[name].append(g)
            ga.append(dict(config=name, seconds=time.perf_counter() - ts, **info))
        audits.append(
            dict(
                cutoff=str(t.date()),
                split=q.split.iloc[0],
                history=len(h),
                queries=q.user_id.nunique(),
                rows=len(q),
                positives=int(q.label.sum()),
                excluded_cold=cold,
            )
        )
        print(audits[-1], flush=True)
    pd.concat(qlist, ignore_index=True).to_pickle(d / "queries.pkl")
    pd.concat(xlist, ignore_index=True).to_pickle(d / "manual.pkl")
    for k, v in gs.items():
        pd.concat(v, ignore_index=True).to_pickle(d / f"{k}.pkl")
    audit = dict(
        commit=COMMIT,
        seed=2026,
        users=len(selected),
        events=len(e),
        source_rows=len(raw),
        source_users=raw.user_id.nunique(),
        disjoint_previous_users=True,
        history_complete=True,
        duplicate_pairs=int(e.duplicated(["user_id", "item_id"]).sum()),
        source_sha256=hashlib.sha256(z.read_bytes()).hexdigest(),
        prepare_seconds=time.perf_counter() - start,
        snapshots=audits,
        graphs=ga,
    )
    (ROOT / "results/data_audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

import json
import pathlib
import time
import zipfile

import numpy as np
import pandas as pd
from advanced_features import build_features
from tune import load_genres

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    out = ROOT / "results/upgrades/confirmation_data"
    out.mkdir(parents=True, exist_ok=True)
    zpath = ROOT / "data/data_original.zip"
    start = time.perf_counter()
    with zipfile.ZipFile(zpath) as z:
        raw = pd.read_csv(z.open("data_original/interactions.csv"))
        items = pd.read_csv(z.open("data_original/items.csv"))
    excluded = np.load(ROOT / "results/upgrades/excluded_confirmation_users.npy")
    rng = np.random.default_rng(20261023)
    selected = np.sort(
        rng.choice(np.setdiff1d(raw.user_id.unique(), excluded), 5000, replace=False)
    )
    assert not set(selected) & set(excluded)
    np.save(out / "users.npy", selected)
    e = raw[raw.user_id.isin(selected)].copy()
    e["date"] = pd.to_datetime(e.last_watch_dt)
    items["genre"] = items.genres.fillna("unknown").str.split(",").str[0].str.strip()
    meta = items[["item_id", "genre", "content_type", "age_rating"]].copy()
    for c in meta.columns[1:]:
        meta[c] = meta[c].fillna("unknown").astype(str)
    e = (
        e.merge(meta, on="item_id", validate="many_to_one")
        .sort_values(["date", "user_id", "item_id"])
        .reset_index(drop=True)
    )
    assert len(e) == raw.user_id.isin(selected).sum()
    e.to_pickle(out / "events.pkl")
    genre = load_genres(zpath)
    end = e.date.max() + pd.Timedelta(days=1)
    frames = []
    features = []
    audit = []
    for j in range(10):
        t = end - pd.Timedelta(days=7 * (10 - j))
        h = e[e.date < t]
        f = e[(e.date >= t) & (e.date < t + pd.Timedelta(days=7))]
        catalog = np.sort(h.item_id.unique())
        catset = set(catalog)
        p = h.item_id.value_counts().reindex(catalog).to_numpy() ** 0.75
        p = p / p.sum()
        rows = []
        for uid, g in f.groupby("user_id", sort=True):
            pos = set(g.item_id) & catset
            if not pos:
                continue
            neg = set()
            for n, probs in [(50, None), (49, p)]:
                target = len(neg) + n
                while len(neg) < target:
                    item = int(rng.choice(catalog, p=probs))
                    if item not in pos:
                        neg.add(item)
            rows.extend((j, uid, i, int(i in pos)) for i in sorted(pos | neg))
        q = pd.DataFrame(rows, columns=["snapshot", "user_id", "item_id", "label"]).merge(
            meta, on="item_id", validate="many_to_one"
        )
        q["cutoff"] = t
        q["split"] = "train" if j < 8 else ("val" if j == 8 else "test")
        features.append(build_features(e, q, t, genre))
        frames.append(q)
        audit.append(
            dict(snapshot=j, cutoff=str(t.date()), rows=len(q), queries=q.user_id.nunique())
        )
        print("fresh", audit[-1], flush=True)
    pd.concat(frames, ignore_index=True).to_pickle(out / "queries.pkl")
    pd.concat(features, ignore_index=True).to_pickle(out / "features.pkl")
    (out / "audit.json").write_text(
        json.dumps(
            dict(
                users=5000,
                excluded_users=len(excluded),
                events=len(e),
                seed=20261023,
                overlap=0,
                seconds=time.perf_counter() - start,
                snapshots=audit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

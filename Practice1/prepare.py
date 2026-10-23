import argparse
import hashlib
import json
import pathlib
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from features import feature_matrix

ROOT = pathlib.Path(__file__).resolve().parent
COMMIT = "f69775be31fa5779907cf0a92ddedb70037fb5ae"


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def download():
    d = ROOT / "data"
    d.mkdir(exist_ok=True)
    for name in ("interactions.csv.zip", "data_original.zip"):
        p = d / name
        if not p.exists() or not zipfile.is_zipfile(p):
            urllib.request.urlretrieve(
                f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{COMMIT}/{name}", p
            )
    return d


def read_csv_zip(path, suffix):
    with zipfile.ZipFile(path) as z:
        n = next(
            n
            for n in z.namelist()
            if not n.startswith("__MACOSX/") and pathlib.PurePosixPath(n).name == suffix
        )
        with z.open(n) as f:
            return pd.read_csv(f)


def build(users=10000, seed=42):
    import time

    start = time.perf_counter()
    d = download()
    rng = np.random.default_rng(seed)

    raw = read_csv_zip(d / "data_original.zip", "interactions.csv")
    all_users = np.sort(raw.user_id.unique())
    selected = np.sort(rng.choice(all_users, min(users, len(all_users)), replace=False))
    e = raw[raw.user_id.isin(selected)].copy()

    e["date"] = pd.to_datetime(e.last_watch_dt)
    items = read_csv_zip(d / "data_original.zip", "items.csv")
    assert not items.item_id.duplicated().any()
    items["genre"] = items.genres.fillna("unknown").str.split(",").str[0].str.strip()
    for c in ("genre", "content_type", "age_rating"):
        items[c] = items[c].fillna("unknown").astype(str)
    meta = items[["item_id", "genre", "content_type", "age_rating"]]
    e = e.merge(meta, on="item_id", how="left", validate="many_to_one")
    for c in ("genre", "content_type", "age_rating"):
        e[c] = e[c].fillna("unknown")
    e = e.sort_values(["date", "user_id", "item_id"]).reset_index(drop=True)
    e.to_pickle(d / "events.pkl")
    np.save(d / "sampled_users.npy", selected)
    end = e.date.max() + pd.Timedelta(days=1)
    cuts = [end - pd.Timedelta(days=7 * i) for i in range(10, 0, -1)]
    frames = []
    feats = []
    audit = []
    for j, t in enumerate(cuts):
        h = e[e.date < t]
        future = e[(e.date >= t) & (e.date < t + pd.Timedelta(days=7))]
        catalog = np.sort(h.item_id.unique())
        catalog_set = set(catalog)
        counts = h.item_id.value_counts().reindex(catalog).to_numpy(dtype=float) ** 0.75
        probs = counts / counts.sum()
        rows = []
        dropped = 0
        for uid, g in future.groupby("user_id", sort=True):
            pos = sorted(set(g.item_id) & catalog_set)
            dropped += len(set(g.item_id) - catalog_set)
            if not pos:
                continue
            excluded = set(pos)
            neg = set()

            for n, p in ((50, None), (49, probs)):
                target = len(neg) + n
                while len(neg) < target:
                    x = int(rng.choice(catalog, p=p))
                    if x not in excluded:
                        neg.add(x)
            for iid in sorted(set(pos) | neg):
                rows.append((j, uid, iid, int(iid in excluded)))
        q = pd.DataFrame(rows, columns=["snapshot", "user_id", "item_id", "label"])
        q = q.merge(meta, on="item_id", how="left", validate="many_to_one")
        for c in ("genre", "content_type", "age_rating"):
            q[c] = q[c].fillna("unknown")
        q["cutoff"] = t
        q["split"] = "train" if j < 8 else ("val" if j == 8 else "test")
        x = feature_matrix(e, q, t, windows=(7, 30, 2, 5, 10, 21, 60), halves=(3, 10, 21))
        frames.append(q)
        feats.append(x)
        audit.append(
            dict(
                snapshot=j,
                cutoff=str(t.date()),
                split=q.split.iloc[0],
                history_rows=len(h),
                history_max=str(h.date.max().date()),
                queries=q.user_id.nunique(),
                rows=len(q),
                positives=int(q.label.sum()),
                catalog=len(catalog),
                excluded_cold_positives=dropped,
            )
        )
        print(audit[-1], flush=True)
    q = pd.concat(frames, ignore_index=True)
    x = pd.concat(feats, ignore_index=True)
    q.to_pickle(d / "queries.pkl")
    x.to_pickle(d / "features.pkl")
    provenance = dict(
        commit=COMMIT,
        seed=seed,
        sampled_users=len(selected),
        source_users=len(all_users),
        source_rows=len(raw),
        sampled_events=len(e),
        duplicate_rows=int(raw.duplicated().sum()),
        sample_duplicate_rows=int(e.duplicated().sum()),
        min_date=str(e.date.min().date()),
        max_date=str(e.date.max().date()),
        input_nulls={k: int(v) for k, v in raw.isna().sum().items()},
        feature_columns=list(x),
        feature_seconds=time.perf_counter() - start,
        sources={n: digest(d / n) for n in ("interactions.csv.zip", "data_original.zip")},
        snapshots=audit,
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_audit.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf8"
    )
    return q, x


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build(a.users, a.seed)

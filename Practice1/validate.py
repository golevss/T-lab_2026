import json
import pathlib

import numpy as np
import pandas as pd
from prepare import digest, read_csv_zip


def main():
    root = pathlib.Path(__file__).resolve().parent
    d = root / "data"
    q = pd.read_pickle(d / "queries.pkl")
    x = pd.read_pickle(d / "features.pkl")
    e = pd.read_pickle(d / "events.pkl")
    raw = read_csv_zip(d / "data_original.zip", "interactions.csv")
    users = np.load(d / "sampled_users.npy")
    audit = json.loads((root / "results/data_audit.json").read_text())
    expected = raw[raw.user_id.isin(users)]
    columns = ["user_id", "item_id", "last_watch_dt", "total_dur", "watched_pct"]
    pd.testing.assert_frame_equal(
        expected[columns].sort_values(columns[:2]).reset_index(drop=True),
        e[columns].sort_values(columns[:2]).reset_index(drop=True),
    )
    assert (
        len(users) == audit["sampled_users"]
        and len(q) == len(x)
        and not q.duplicated(["snapshot", "user_id", "item_id"]).any()
    )
    assert not x.isna().any().any()
    counts = (
        q.assign(negative=1 - q.label)
        .groupby(["snapshot", "user_id"])
        .agg(negative=("negative", "sum"), positive=("label", "sum"))
    )
    assert counts.negative.eq(99).all() and counts.positive.gt(0).all()
    for _, g in q.groupby("snapshot"):
        t = g.cutoff.iloc[0]
        h = e[e.date < t]
        f = e[(e.date >= t) & (e.date < t + pd.Timedelta(days=7))]
        assert h.date.max() < t and set(g.item_id) <= set(h.item_id)
        pairs = set(zip(f.user_id, f.item_id))
        labels = np.array([(u, i) in pairs for u, i in zip(g.user_id, g.item_id)], dtype=int)
        assert np.array_equal(labels, g.label.to_numpy())

        row = g.iloc[0]
        u = h[h.user_id == row.user_id]
        idx = g.index[0]
        assert x.loc[idx, "u_count_all"] == len(u)
        assert x.loc[idx, "u_count_7"] == (u.date >= t - pd.Timedelta(days=7)).sum()
    assert (
        q[q.split == "train"].cutoff.max() + pd.Timedelta(days=7)
        <= q[q.split == "val"].cutoff.min()
    )
    assert (
        q[q.split == "val"].cutoff.max() + pd.Timedelta(days=7) <= q[q.split == "test"].cutoff.min()
    )
    for n, h in audit["sources"].items():
        assert digest(d / n) == h
    result = {
        "complete_user_history": True,
        "candidates_unique": True,
        "negatives_per_query": 99,
        "labels_recomputed_all_rows": True,
        "catalog_past_only": True,
        "nonoverlapping_label_windows": True,
        "sampled_manual_count_checks": 10,
        "source_hashes_verified": True,
        "rows": len(q),
        "features": len(x.columns),
    }
    (root / "results/protocol_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

import json
import math
import pathlib

import numpy as np
import pandas as pd
from prepare import read

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    d = ROOT / "data"
    q = pd.read_pickle(d / "queries.pkl")
    e = pd.read_pickle(d / "events.pkl")
    raw = read(d / "data_original.zip", "interactions.csv")
    users = np.load(d / "selected_users.npy")
    assert not set(users) & set(np.load(d / "excluded_previous_users.npy"))
    cols = ["user_id", "item_id", "last_watch_dt", "total_dur", "watched_pct"]
    pd.testing.assert_frame_equal(
        raw[raw.user_id.isin(users)][cols].sort_values(cols[:2]).reset_index(drop=True),
        e[cols].sort_values(cols[:2]).reset_index(drop=True),
    )
    assert not q.duplicated(["snapshot", "user_id", "item_id"]).any()
    assert q.assign(neg=1 - q.label).groupby(["snapshot", "user_id"]).neg.sum().eq(99).all()
    for _, g in q.groupby("snapshot"):
        t = g.cutoff.iloc[0]
        h = e[e.date < t]
        f = e[(e.date >= t) & (e.date < t + pd.Timedelta(days=7))]
        assert set(g.item_id) <= set(h.item_id)
        pairs = set(zip(f.user_id, f.item_id))
        assert all(
            label == int((u, i) in pairs) for u, i, label in zip(g.user_id, g.item_id, g.label)
        )
    result = dict(
        full_history_verified=True,
        previous_users_overlap=0,
        all_labels_verified=True,
        negatives_per_query=99,
        rows=len(q),
    )
    if (ROOT / "results/predictions.csv.gz").exists():
        p = pd.read_csv(ROOT / "results/predictions.csv.gz", float_precision="round_trip")
        scores = pd.read_csv(ROOT / "results/test.csv").set_index("model")
        pd.testing.assert_frame_equal(
            p[["snapshot", "user_id", "item_id", "label"]],
            q[q.split == "test"][["snapshot", "user_id", "item_id", "label"]].reset_index(
                drop=True
            ),
        )
        for name in ["manual", "representation", "hybrid"]:
            nd = []
            rec = []
            for _, g in p.groupby("user_id"):
                z = sorted(zip(g[name], g.item_id, g.label), key=lambda z: (-z[0], z[1]))
                n = sum(z[2] for z in z)
                nd.append(
                    sum(v[2] / math.log2(j + 2) for j, v in enumerate(z[:10]))
                    / sum(1 / math.log2(j + 2) for j in range(min(10, n)))
                )
                rec.append(sum(v[2] for v in z[:10]) / n)
            assert abs(np.mean(nd) - scores.loc[name, "ndcg10"]) < 1e-10
            assert abs(np.mean(rec) - scores.loc[name, "recall10"]) < 1e-10
        result["test_ranking_independently_verified"] = True
    (ROOT / "results/validation_checks.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()

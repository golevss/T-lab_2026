import json
import math
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def main():
    root = pathlib.Path(__file__).resolve().parent
    r = root / "results"
    p = pd.read_csv(r / "test_predictions.csv.gz", float_precision="round_trip")
    reported = pd.read_csv(r / "test_metrics.csv").set_index("model")
    q = pd.read_pickle(root / "data/queries.pkl")
    q = q[q.split == "test"]
    cols = ["snapshot", "user_id", "item_id", "label"]
    pd.testing.assert_frame_equal(p[cols].reset_index(drop=True), q[cols].reset_index(drop=True))
    records = []
    for name in ("manual", "automatic"):
        ndcgs = []
        recalls = []
        for _, g in p.groupby(["snapshot", "user_id"]):
            rows = sorted(zip(g[name], g.item_id, g.label), key=lambda z: (-z[0], z[1]))
            total = sum(z[2] for z in rows)
            dcg = sum(row[2] / math.log2(j + 2) for j, row in enumerate(rows[:10]))
            ideal = sum(1 / math.log2(j + 2) for j in range(min(10, total)))
            ndcgs.append(dcg / ideal)
            recalls.append(sum(z[2] for z in rows[:10]) / total)
        values = dict(
            logloss=log_loss(p.label, p[name]),
            roc_auc=roc_auc_score(p.label, p[name]),
            ndcg10=np.mean(ndcgs),
            recall10=np.mean(recalls),
        )
        for key, val in values.items():
            assert abs(reported.loc[name, key] - val) < 1e-10, (name, key, val)
        records.append({"model": name, **{k: float(v) for k, v in values.items()}})
    (r / "result_validation.json").write_text(
        json.dumps({"all_metrics_recomputed": True, "models": records}, indent=2)
    )
    print("All final model metrics match independently recomputed ranking and exported scores.")


if __name__ == "__main__":
    main()

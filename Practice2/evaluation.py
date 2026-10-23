import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def ranking(q, scores, k=10):
    z = q[["snapshot", "user_id", "item_id", "label"]].copy()
    z["score"] = scores

    z = z.sort_values(
        ["snapshot", "user_id", "score", "item_id"], ascending=[True, True, False, True]
    )
    z["rank"] = z.groupby(["snapshot", "user_id"]).cumcount() + 1
    z["gain"] = np.where(z["rank"] <= k, z.label / np.log2(z["rank"] + 1), 0)
    z["hit"] = np.where(z["rank"] <= k, z.label, 0)
    g = z.groupby(["snapshot", "user_id"]).agg(
        dcg=("gain", "sum"), hits=("hit", "sum"), positives=("label", "sum")
    )
    ideal = np.array([sum(1 / np.log2(np.arange(1, min(k, int(n)) + 1) + 1)) for n in g.positives])
    g["ndcg"] = g.dcg / ideal
    g["recall"] = g.hits / g.positives
    return g


def metrics(q, scores):
    r = ranking(q, scores)
    return dict(
        logloss=float(log_loss(q.label, np.clip(scores, 1e-9, 1 - 1e-9), labels=[0, 1])),
        roc_auc=float(roc_auc_score(q.label, scores)),
        ndcg10=float(r.ndcg.mean()),
        recall10=float(r.recall.mean()),
        queries=len(r),
    )

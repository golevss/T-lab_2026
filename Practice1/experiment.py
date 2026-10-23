import json
import pathlib
import platform
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from features import feature_bundles, manual_columns
from sklearn.metrics import log_loss, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent
PARAMS = [
    dict(depth=5, iterations=180, learning_rate=0.08),
    dict(depth=6, iterations=260, learning_rate=0.06),
    dict(depth=7, iterations=320, learning_rate=0.05),
]


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


def fit(x, y, cols, params, seed=42):
    m = CatBoostClassifier(
        **params,
        loss_function="Logloss",
        random_seed=seed,
        thread_count=6,
        verbose=False,
        allow_writing_files=False,
        l2_leaf_reg=5,
    )
    cats = [c for c in ("genre", "content_type", "age_rating") if c in cols]
    start = time.perf_counter()
    m.fit(x[cols], y, cat_features=cats)
    return m, time.perf_counter() - start


def main():
    start = time.perf_counter()
    d = ROOT / "data"
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    q = pd.read_pickle(d / "queries.pkl")
    x = pd.read_pickle(d / "features.pkl")
    tr = q.split == "train"
    va = q.split == "val"
    te = q.split == "test"

    train_keys = q.loc[tr, ["snapshot", "user_id"]].drop_duplicates()
    if len(train_keys) > 5000:
        chosen_keys = train_keys.sample(n=5000, random_state=42)
        key_index = pd.MultiIndex.from_frame(chosen_keys)
        tr = tr & pd.MultiIndex.from_frame(q[["snapshot", "user_id"]]).isin(key_index)
    train_query_count = q.loc[tr, ["snapshot", "user_id"]].drop_duplicates().shape[0]
    manual = manual_columns(x.columns)
    bundles = feature_bundles(x.columns)
    trials = []

    def trial(name, cols, params):
        m, secs = fit(x[tr], q.loc[tr, "label"], cols, params)
        met = metrics(q[va], m.predict_proba(x.loc[va, cols])[:, 1])
        row = dict(name=name, columns=len(cols), parameters=params, fit_seconds=secs, **met)
        trials.append(row)
        print(json.dumps(row), flush=True)
        return met["ndcg10"]

    base_scores = [trial(f"manual_grid_{j}", manual, p) for j, p in enumerate(PARAMS)]
    best_base = int(np.argmax(base_scores))
    chosen = []
    cols = manual.copy()
    incumbent = base_scores[0]

    for step in range(2):
        candidates = []
        for name, cs in bundles.items():
            if name in chosen:
                continue
            score = trial(f"search_{step}_{name}", cols + cs, PARAMS[0])
            candidates.append((score, name))
        score, name = max(candidates)
        if score <= incumbent + 1e-5:
            break
        chosen.append(name)
        cols += bundles[name]
        incumbent = score
    auto_scores = [trial(f"auto_grid_{j}", cols, p) for j, p in enumerate(PARAMS)]
    best_auto = int(np.argmax(auto_scores))

    trial("all_generated", list(x.columns), PARAMS[0])
    trial(
        "manual_without_indirect",
        [c for c in manual if not c.startswith(("ug_", "uc_"))],
        PARAMS[0],
    )
    trial("manual_without_direct", [c for c in manual if not c.startswith("ui_")], PARAMS[0])
    trial(
        "automatic_only",
        [c for c in x if c not in manual] + ["genre", "content_type", "age_rating"],
        PARAMS[0],
    )
    pd.DataFrame(trials).to_json(out / "validation_trials.json", orient="records", indent=2)
    frozen = dict(
        manual_columns=manual,
        selected_bundles=chosen,
        auto_columns=cols,
        manual_params=PARAMS[best_base],
        auto_params=PARAMS[best_auto],
    )
    (out / "frozen_selection.json").write_text(json.dumps(frozen, indent=2))

    dev = tr | va
    final = []
    predictions = q.loc[te, ["snapshot", "user_id", "item_id", "label"]].copy()
    for name, cs, p in [
        ("manual", manual, PARAMS[best_base]),
        ("automatic", cols, PARAMS[best_auto]),
    ]:
        m, secs = fit(x[dev], q.loc[dev, "label"], cs, p)
        t = time.perf_counter()
        pred = m.predict_proba(x.loc[te, cs])[:, 1]
        pred_secs = time.perf_counter() - t
        predictions[name] = pred
        m.save_model(str(out / f"{name}.cbm"))
        final.append(
            dict(
                model=name,
                features=len(cs),
                fit_seconds=secs,
                predict_seconds=pred_secs,
                **metrics(q[te], pred),
            )
        )
        pd.DataFrame({"feature": cs, "importance": m.feature_importances_}).sort_values(
            "importance", ascending=False
        ).to_csv(out / f"{name}_importance.csv", index=False)

    pop = x.loc[te, "i_count_30"].to_numpy()
    p = pop / (pop.max() + 1) * 0.1 + 1e-5
    final.append(
        dict(
            model="popularity30", features=1, fit_seconds=0, predict_seconds=0, **metrics(q[te], p)
        )
    )

    final[-1]["logloss"] = None
    final[-1]["roc_auc"] = float(roc_auc_score(q.loc[te, "label"], pop))
    predictions.to_csv(out / "test_predictions.csv.gz", index=False)
    pd.DataFrame(final).to_csv(out / "test_metrics.csv", index=False)
    a = ranking(q[te], predictions.manual)
    b = ranking(q[te], predictions.automatic)
    delta = (b.ndcg - a.ndcg).groupby("user_id").mean().to_numpy()
    rng = np.random.default_rng(2026)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
    ci = dict(
        delta_ndcg=float(delta.mean()),
        paired_user_bootstrap_95=np.quantile(boot, [0.025, 0.975]).tolist(),
        users=len(delta),
        resamples=2000,
    )
    runtime = dict(
        total_experiment_seconds=time.perf_counter() - start,
        selection_fit_seconds=sum(r["fit_seconds"] for r in trials),
        train_query_cap=5000,
        training_queries=train_query_count,
        training_rows=int(tr.sum()),
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        pandas=pd.__version__,
        catboost=__import__("catboost").__version__,
        sklearn=__import__("sklearn").__version__,
    )
    (out / "uncertainty.json").write_text(json.dumps(ci, indent=2))
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print(pd.DataFrame(final).to_string(index=False), flush=True)
    print(ci, flush=True)


if __name__ == "__main__":
    main()

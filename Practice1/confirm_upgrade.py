import json
import pathlib

import numpy as np
import pandas as pd
from advanced_features import columns_for
from evaluation import metrics, ranking
from tune import fit_model, predict

ROOT = pathlib.Path(__file__).resolve().parent


def comparison(queries, baseline, candidate):
    delta = ranking(queries, candidate).ndcg.to_numpy() - ranking(queries, baseline).ndcg.to_numpy()
    rng = np.random.default_rng(7351)
    draws = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(4000)]
    return {
        "delta": float(delta.mean()),
        "ci_family95": np.quantile(draws, [0.05 / 6, 1 - 0.05 / 6]).tolist(),
        "queries": len(delta),
    }


def main():
    out = ROOT / "results/upgrades"
    data = out / "confirmation_data"
    frozen = json.loads((out / "selection.json").read_text())
    kind = frozen["kind"]
    q = pd.read_pickle(data / "queries.pkl")
    x = pd.read_pickle(data / "features.pkl")
    dev = (q.split != "test") & (
        (q.snapshot >= 2) if kind == "graph" else np.ones(len(q), dtype=bool)
    )
    test = q.split == "test"
    values = {}
    results = {}
    for name, config in [("baseline", frozen["baseline"]), ("candidate", frozen["winner"])]:
        cols = columns_for(x, config["mode"])
        model = fit_model(x[dev], q[dev], cols, config["params"], config["ranker"])
        scores = predict(model, x[test], cols, config["ranker"])
        values[name] = scores
        results[name] = metrics(q[test], scores)
        if config["ranker"]:
            results[name]["logloss"] = None
    results["comparison"] = comparison(q[test], values["baseline"], values["candidate"])
    if kind == "graph":
        cols = [c for c in x if c.startswith("g_")]
        params = json.loads((ROOT / "results/frozen_selection.json").read_text())["parameters"][
            "representation"
        ]
        rep = fit_model(x[dev], q[dev], cols, params)
        raw = predict(rep, x[test], cols)
        fallback = np.where(x.loc[test, "g_cold"].to_numpy() > 0, values["baseline"], raw)
        results["cold_start_comparison"] = comparison(q[test], raw, fallback)
    expected = json.loads((out / "confirmation.json").read_text())
    for name, index in [("baseline", 0), ("candidate", 1)]:
        for metric in ["ndcg10", "recall10", "roc_auc"]:
            assert abs(results[name][metric] - expected["metrics"][index][metric]) < 1e-10
    (out / "confirmation_reproduced.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

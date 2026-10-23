import json
import pathlib
import platform
import time

import numpy as np
import pandas as pd
from metrics import fit, metrics, ranking

ROOT = pathlib.Path(__file__).resolve().parent
GRID = [
    dict(depth=5, iterations=180, learning_rate=0.08),
    dict(depth=7, iterations=260, learning_rate=0.05),
]


def main():
    start = time.perf_counter()
    d = ROOT / "data"
    r = ROOT / "results"
    q = pd.read_pickle(d / "queries.pkl")
    manual = pd.read_pickle(d / "manual.pkl")
    graphs = {n: pd.read_pickle(d / f"{n}.pkl") for n in ["graph16", "graph32", "no_categories"]}
    tr = q.split == "train"
    va = q.split == "val"
    te = q.split == "test"
    trials = []

    def run(name, x, params):
        m, seconds = fit(x[tr], q.loc[tr, "label"], list(x), params, seed=2026)
        row = dict(
            name=name,
            parameters=params,
            features=len(x.columns),
            fit_seconds=seconds,
            **metrics(q[va], m.predict_proba(x[va])[:, 1]),
        )
        trials.append(row)
        print(row, flush=True)
        return row["ndcg10"]

    base = [run(f"manual_{i}", manual, p) for i, p in enumerate(GRID)]
    selected = {}
    matrices = {"manual": manual}
    params = {"manual": GRID[int(np.argmax(base))]}
    for mode in ["representation", "hybrid"]:
        results = []
        for name, g in graphs.items():
            x = g if mode == "representation" else pd.concat([manual, g], axis=1)
            results.append((run(f"{mode}_{name}", x, GRID[0]), name))

        winner = max(v for v in results if v[1] != "no_categories")[1]
        g = graphs[winner]
        x = g if mode == "representation" else pd.concat([manual, g], axis=1)
        extra = run(f"{mode}_grid1", x, GRID[1])
        first = next(s for s, n in results if n == winner)
        selected[mode] = winner
        matrices[mode] = x
        params[mode] = GRID[int(extra > first)]
    (r / "validation.json").write_text(json.dumps(trials, indent=2))
    (r / "frozen_selection.json").write_text(
        json.dumps(dict(graph=selected, parameters=params), indent=2)
    )
    pred = q.loc[te, ["snapshot", "user_id", "item_id", "label"]].reset_index(drop=True)
    final = []
    dev = tr | va
    for name, x in matrices.items():
        m, secs = fit(x[dev], q.loc[dev, "label"], list(x), params[name], seed=2026)
        ts = time.perf_counter()
        scores = m.predict_proba(x[te])[:, 1]
        ps = time.perf_counter() - ts
        pred[name] = scores
        m.save_model(str(r / f"{name}.cbm"))
        final.append(
            dict(model=name, fit_seconds=secs, predict_seconds=ps, **metrics(q[te], scores))
        )
        pd.DataFrame({"feature": list(x), "importance": m.feature_importances_}).sort_values(
            "importance", ascending=False
        ).to_csv(r / f"{name}_importance.csv", index=False)
    final.append(
        dict(
            model="popularity30",
            fit_seconds=0,
            predict_seconds=0,
            **metrics(
                q[te], manual.loc[te, "i_count_30"].to_numpy() / (manual.i_count_30.max() + 1)
            ),
        )
    )
    final[-1]["logloss"] = None
    pred.to_csv(r / "predictions.csv.gz", index=False)
    pd.DataFrame(final).to_csv(r / "test.csv", index=False)
    intervals = {}
    a = ranking(q[te], pred.manual.to_numpy()).ndcg.to_numpy()
    for name in ["representation", "hybrid"]:
        delta = ranking(q[te], pred[name].to_numpy()).ndcg.to_numpy() - a
        rng = np.random.default_rng(2026)
        boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)]
        intervals[name] = dict(
            delta=float(delta.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist()
        )
    (r / "intervals.json").write_text(json.dumps(intervals, indent=2))
    (r / "runtime.json").write_text(
        json.dumps(
            dict(
                total_seconds=time.perf_counter() - start,
                validation_fit_seconds=sum(t["fit_seconds"] for t in trials),
                platform=platform.platform(),
            ),
            indent=2,
        )
    )
    print(pd.DataFrame(final).to_string(index=False))
    print(intervals)


if __name__ == "__main__":
    main()

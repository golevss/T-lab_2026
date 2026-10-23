import argparse
import json
import pathlib
import time
import zipfile

import pandas as pd
from advanced_features import build_features, columns_for
from catboost import CatBoostClassifier, CatBoostRanker
from evaluation import metrics


def fit_model(x, q, columns, params, ranker=False):
    args = dict(
        **params,
        random_seed=2026,
        thread_count=6,
        verbose=False,
        allow_writing_files=False,
        l2_leaf_reg=5,
    )
    cats = [c for c in ["genre", "content_type", "age_rating"] if c in columns]
    if ranker:
        model = CatBoostRanker(**args, loss_function="YetiRank:mode=NDCG;top=10")
        groups = pd.factorize(pd.MultiIndex.from_frame(q[["snapshot", "user_id"]]), sort=False)[0]
        model.fit(x[columns], q.label, group_id=groups, cat_features=cats)
    else:
        model = CatBoostClassifier(**args, loss_function="Logloss")
        model.fit(x[columns], q.label, cat_features=cats)
    return model


def predict(model, x, columns, ranker=False):
    return model.predict(x[columns]) if ranker else model.predict_proba(x[columns])[:, 1]


def load_genres(path):
    with zipfile.ZipFile(path) as z:
        name = next(
            n for n in z.namelist() if n.endswith("/items.csv") and not n.startswith("__MACOSX/")
        )
        items = pd.read_csv(z.open(name))
    return {
        row.item_id: [g.strip() for g in str(row.genres).split(",")]
        for row in items.fillna({"genres": "unknown"}).itertuples()
    }


def run_search(project, kind):
    out = project / "results/upgrades"
    out.mkdir(exist_ok=True)
    q = pd.read_pickle(project / "data/queries.pkl")
    e = pd.read_pickle(project / "data/events.pkl")
    cache = out / "features.pkl"
    from train_active import feature_key

    key = feature_key(e, q, project / "data/data_original.zip")
    key_path = out / "features_key.json"
    if cache.exists() and key_path.exists() and json.loads(key_path.read_text())["key"] == key:
        x = pd.read_pickle(cache)
    else:
        genre = load_genres(project / "data/data_original.zip")
        frames = []
        for snapshot, frame in q.groupby("snapshot", sort=True):
            frames.append(build_features(e, frame, frame.cutoff.iloc[0], genre))
            print(kind, "features", snapshot, flush=True)
        x = pd.concat(frames, ignore_index=True)
        x.to_pickle(cache)
        key_path.write_text(json.dumps({"key": key, "version": 1}))
    training = q.split == "train"
    if kind == "features":
        keys = q.loc[training, ["snapshot", "user_id"]].drop_duplicates()
        selected = keys.sample(n=min(5000, len(keys)), random_state=42)
        training &= pd.MultiIndex.from_frame(q[["snapshot", "user_id"]]).isin(
            pd.MultiIndex.from_frame(selected)
        )
    keys = q.loc[training, ["snapshot", "user_id"]].drop_duplicates()
    selected = keys.sample(n=min(1500, len(keys)), random_state=9102)
    screen = training & pd.MultiIndex.from_frame(q[["snapshot", "user_id"]]).isin(
        pd.MultiIndex.from_frame(selected)
    )
    val = q.split == "val"
    base_mode = "manual" if kind == "features" else "graph"
    screen_params = dict(depth=5, iterations=180, learning_rate=0.08)
    trials = []
    variants = [
        ("baseline", base_mode, False),
        ("item_knn", "knn", False),
        ("rich", "rich", False),
        ("ranker", "rich", True),
    ]
    if kind == "features":
        variants.insert(1, ("spectral", "graph", False))
    for name, mode, ranker in variants:
        params = dict(depth=5, iterations=120, learning_rate=0.08) if ranker else screen_params
        columns = columns_for(x, mode)
        start = time.perf_counter()
        model = fit_model(x[screen], q[screen], columns, params, ranker)
        result = metrics(q[val], predict(model, x[val], columns, ranker))
        if ranker:
            result["logloss"] = None
        trial = dict(
            name=name,
            mode=mode,
            ranker=ranker,
            params=params,
            features=len(columns),
            fit_seconds=time.perf_counter() - start,
            **result,
        )
        trials.append(trial)
        print(kind, trial, flush=True)
    winner = max(trials[1:], key=lambda t: t["ndcg10"])
    original = json.loads((project / "results/frozen_selection.json").read_text())
    base_params = (
        original["manual_params"] if kind == "features" else original["parameters"]["hybrid"]
    )
    final = []
    for label, mode, params, ranker in [
        ("baseline", base_mode, base_params, False),
        (
            "candidate",
            winner["mode"],
            winner["params"] if winner["ranker"] else base_params,
            winner["ranker"],
        ),
    ]:
        columns = columns_for(x, mode)
        start = time.perf_counter()
        model = fit_model(x[training], q[training], columns, params, ranker)
        result = metrics(q[val], predict(model, x[val], columns, ranker))
        if ranker:
            result["logloss"] = None
        final.append(
            dict(
                name=label,
                mode=mode,
                params=params,
                ranker=ranker,
                features=len(columns),
                seconds=time.perf_counter() - start,
                **result,
            )
        )
        print(kind, "full", final[-1], flush=True)
    frozen = dict(kind=kind, trials=trials, validation=final, winner=final[1], baseline=final[0])
    (out / "selection.json").write_text(json.dumps(frozen, indent=2))
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=pathlib.Path, required=True)
    parser.add_argument("--kind", choices=["features", "graph"], required=True)
    args = parser.parse_args()
    run_search(args.project, args.kind)

import hashlib
import json
import pathlib

import pandas as pd
from advanced_features import build_features
from tune import fit_model, load_genres

ROOT = pathlib.Path(__file__).resolve().parent


def feature_key(events, queries, source=None):
    h = hashlib.sha256()
    for frame in [events, queries[["snapshot", "user_id", "item_id", "cutoff"]]]:
        h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    if source is not None:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def load_features(root, events, queries):
    out = root / "results/upgrades"
    key = feature_key(events, queries, root / "data/data_original.zip")
    metadata = out / "features_key.json"
    cache = out / "features.pkl"
    if cache.exists() and metadata.exists() and json.loads(metadata.read_text())["key"] == key:
        return pd.read_pickle(cache)
    genre = load_genres(root / "data/data_original.zip")
    frames = []
    for _, q in queries.groupby("snapshot", sort=True):
        frames.append(build_features(events, q, q.cutoff.iloc[0], genre))
    x = pd.concat(frames, ignore_index=True)
    x.to_pickle(cache)
    metadata.write_text(json.dumps({"key": key, "version": 1}))
    return x


def main():
    config = json.loads((ROOT / "active_model.json").read_text())
    if not config["promoted"]:
        print("Using the existing model")
        return
    q = pd.read_pickle(ROOT / "data/queries.pkl")
    e = pd.read_pickle(ROOT / "data/events.pkl")
    x = load_features(ROOT, e, q)
    train = q.split == "train"
    if config["project"] == "features":
        keys = q.loc[train, ["snapshot", "user_id"]].drop_duplicates()
        chosen = keys.sample(n=min(5000, len(keys)), random_state=42)
        train &= pd.MultiIndex.from_frame(q[["snapshot", "user_id"]]).isin(
            pd.MultiIndex.from_frame(chosen)
        )
    dev = train | (q.split == "val")
    model = fit_model(x[dev], q[dev], config["columns"], config["params"], config["ranker"])
    model.save_model(str(ROOT / config["model"]))
    print("Active model trained:", config["model"])


if __name__ == "__main__":
    main()

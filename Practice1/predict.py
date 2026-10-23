import argparse
import json
import pathlib
import zipfile

import numpy as np
import pandas as pd
from advanced_features import build_features
from catboost import CatBoostClassifier, CatBoostRanker
from tune import load_genres

ROOT = pathlib.Path(__file__).resolve().parent


def score(events, candidates, cutoff, mode="active"):
    config = json.loads((ROOT / "active_model.json").read_text())
    cutoff = pd.Timestamp(cutoff)
    history = events[events.date < cutoff]
    if not set(candidates.item_id) <= set(history.item_id):
        raise ValueError("Candidates must belong to the historical catalog")
    with zipfile.ZipFile(ROOT / "data/data_original.zip") as z:
        name = next(
            n for n in z.namelist() if n.endswith("/items.csv") and not n.startswith("__MACOSX/")
        )
        meta = pd.read_csv(z.open(name))
    meta["genre"] = meta.genres.fillna("unknown").str.split(",").str[0].str.strip()
    meta = meta[["item_id", "genre", "content_type", "age_rating"]].copy()
    for c in meta.columns[1:]:
        meta[c] = meta[c].fillna("unknown").astype(str)
    q = candidates[["user_id", "item_id"]].merge(meta, on="item_id", validate="many_to_one")
    if mode not in {"active", "representation", "representation_raw"}:
        raise ValueError("Unknown prediction mode")
    if mode != "active" and config["project"] != "graph":
        raise ValueError("Representation mode exists only in the graph project")
    feature_mode = config["mode"] if mode == "active" else "graph"
    x = build_features(
        events, q, cutoff, load_genres(ROOT / "data/data_original.zip"), mode=feature_mode
    )
    if mode == "active":
        model = CatBoostRanker() if config["ranker"] else CatBoostClassifier()
        model.load_model(str(ROOT / config["model"]))
        active = (
            model.predict(x[config["columns"]])
            if config["ranker"]
            else model.predict_proba(x[config["columns"]])[:, 1]
        )
    else:
        raw_model = CatBoostClassifier()
        raw_model.load_model(str(ROOT / "results/representation.cbm"))
        raw = raw_model.predict_proba(x[raw_model.feature_names_])[:, 1]
        if mode == "representation" and config.get("cold_start_fallback", False):
            fallback = CatBoostClassifier()
            fallback.load_model(str(ROOT / "results/hybrid.cbm"))
            fallback_scores = fallback.predict_proba(x[fallback.feature_names_])[:, 1]
            active = np.where(x.g_cold.to_numpy() > 0, fallback_scores, raw)
        else:
            active = raw
    out = q[["user_id", "item_id"]].copy()
    out["score"] = active
    return out.sort_values(
        ["user_id", "score", "item_id"], ascending=[True, False, True]
    ).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=pathlib.Path, default=ROOT / "data/events.pkl")
    parser.add_argument("--candidates", type=pathlib.Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--mode", choices=["active", "representation", "representation_raw"], default="active"
    )
    args = parser.parse_args()
    score(pd.read_pickle(args.events), pd.read_csv(args.candidates), args.cutoff, args.mode).to_csv(
        args.output, index=False
    )

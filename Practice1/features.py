import numpy as np
import pandas as pd

KEYS = {
    "u": ["user_id"],
    "i": ["item_id"],
    "ui": ["user_id", "item_id"],
    "ug": ["user_id", "genre"],
    "uc": ["user_id", "content_type"],
}


def feature_matrix(events, candidates, cutoff, windows=(7, 30), halves=()):

    h = events.loc[events.date < cutoff].copy()
    q = candidates.copy().reset_index(drop=True)
    out = q[["genre", "content_type", "age_rating"]].fillna("unknown").astype(str)
    age = (cutoff - h.date).dt.days.astype(float)

    def mapped(frame, keys, values):
        s = frame.assign(_v=values).groupby(keys, observed=True)._v.sum()
        idx = pd.MultiIndex.from_frame(q[keys]) if len(keys) > 1 else q[keys[0]]
        return s.reindex(idx).fillna(0).to_numpy(dtype=np.float32)

    for tag, keys in KEYS.items():
        out[f"{tag}_count_all"] = mapped(h, keys, np.ones(len(h)))
        last = h.groupby(keys, observed=True).date.max()
        idx = pd.MultiIndex.from_frame(q[keys]) if len(keys) > 1 else q[keys[0]]
        dates = last.reindex(idx).to_numpy()
        out[f"{tag}_recency"] = np.nan_to_num(
            (cutoff.to_datetime64() - dates) / np.timedelta64(1, "D"), nan=9999
        ).astype("float32")
        for w in windows:
            hw = h.loc[age <= w]
            out[f"{tag}_count_{w}"] = mapped(hw, keys, np.ones(len(hw)))
        for half in halves:
            out[f"{tag}_decay_{half}"] = mapped(h, keys, np.exp2(-age / half))
    for w in ("all", *windows):
        out[f"ug_share_{w}"] = out[f"ug_count_{w}"] / (out[f"u_count_{w}"] + 1)
        out[f"uc_share_{w}"] = out[f"uc_count_{w}"] / (out[f"u_count_{w}"] + 1)
        if w != "all":
            out[f"i_trend_{w}"] = out[f"i_count_{w}"] / (out.i_count_all + 1)

    for tag in ("u", "i", "ug"):
        keys = KEYS[tag]
        out[f"{tag}_completion_mean"] = mapped(h, keys, h.watched_pct.fillna(0)) / (
            out[f"{tag}_count_all"] + 1
        )
    assert len(out) == len(q) and not out.isna().any().any()
    return out


def manual_columns(all_columns):
    return [
        c
        for c in all_columns
        if not any(f"_{w}" == c[-len(str(w)) - 1 :] for w in (2, 5, 10, 21, 60))
        and "_decay_" not in c
    ]


def feature_bundles(columns):
    return {
        **{
            f"window_{w}": [c for c in columns if c.endswith(f"_{w}") and "_decay_" not in c]
            for w in (2, 5, 10, 21, 60)
        },
        **{f"decay_{h}": [c for c in columns if c.endswith(f"_decay_{h}")] for h in (3, 10, 21)},
    }

import unittest

import numpy as np
import pandas as pd
from advanced_features import build_features, columns_for


class UpgradesTests(unittest.TestCase):
    def test_future_and_cold_history(self):
        t = pd.Timestamp("2021-01-10")
        e = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 1],
                "item_id": [10, 11, 11, 12, 12, 12],
                "date": pd.to_datetime(
                    [
                        "2021-01-01",
                        "2021-01-02",
                        "2021-01-02",
                        "2021-01-03",
                        "2021-01-03",
                        "2021-01-10",
                    ]
                ),
                "genre": ["a"] * 6,
                "content_type": ["film"] * 6,
                "age_rating": ["12"] * 6,
                "watched_pct": [50] * 6,
            }
        )
        q = e.iloc[[0, 2]].reset_index(drop=True)
        q.loc[1, "user_id"] = 999
        genres = {10: ["a", "b"], 11: ["b"], 12: ["c"]}
        a = build_features(e, q, t, genres)
        for mode in ["manual", "graph", "knn"]:
            pd.testing.assert_frame_equal(
                a[columns_for(a, mode)], build_features(e, q, t, genres, mode=mode)
            )
        future = e[e.date >= t].copy()
        future.watched_pct = 10000
        changed = pd.concat([e[e.date < t], future, future], ignore_index=True)
        b = build_features(changed, q, t, genres)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue((a.loc[1, ["knn_all", "knn_7", "knn_last"]] == 0).all())
        self.assertGreater(a.loc[0, "genres_max_all"], 0)
        self.assertTrue(np.isfinite(a.select_dtypes("number").to_numpy()).all())


if __name__ == "__main__":
    unittest.main()

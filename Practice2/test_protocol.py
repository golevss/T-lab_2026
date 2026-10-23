import unittest

import numpy as np
import pandas as pd
from features import feature_matrix
from graph import graph_features


class Protocol(unittest.TestCase):
    def test_future_invariance(self):
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
        q = e.iloc[[0, 2]].copy().reset_index(drop=True)
        q.loc[1, "user_id"] = 999
        future = e[e.date >= t].copy()
        future.watched_pct = 9999
        altered = pd.concat([e[e.date < t], future, future], ignore_index=True)
        for cats in [True, False]:
            a, _ = graph_features(e, q, t, 2, cats)
            b, _ = graph_features(altered, q, t, 2, cats)
            np.testing.assert_allclose(a, b, atol=1e-6)
            self.assertEqual(a.loc[1, "g_cold"], 1)
        pd.testing.assert_frame_equal(feature_matrix(e, q, t), feature_matrix(altered, q, t))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import pandas as pd
from experiment import ranking
from features import feature_bundles, feature_matrix, manual_columns


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.t = pd.Timestamp("2021-06-10")
        self.e = pd.DataFrame(
            dict(
                user_id=[1, 1, 1, 2],
                item_id=[10, 11, 10, 10],
                date=pd.to_datetime(["2021-06-03", "2021-06-09", "2021-06-10", "2021-06-11"]),
                genre=["A"] * 4,
                content_type=["film"] * 4,
                age_rating=["12"] * 4,
                watched_pct=[20, 80, 100, 100],
            )
        )
        self.q = self.e.iloc[[0, 3]].drop(columns=["date", "watched_pct"]).reset_index(drop=True)

    def test_future_and_target_invariance(self):
        a = feature_matrix(
            self.e, self.q, self.t, windows=(7, 30, 2, 5, 10, 21, 60), halves=(3, 10, 21)
        )
        b = self.e.copy()
        b.loc[b.date >= self.t, "watched_pct"] = 99999
        b = pd.concat([b, b[b.date >= self.t]] * 2, ignore_index=True)

        b = pd.concat([self.e[self.e.date < self.t], b[b.date >= self.t]], ignore_index=True)
        pd.testing.assert_frame_equal(
            a,
            feature_matrix(
                b, self.q, self.t, windows=(7, 30, 2, 5, 10, 21, 60), halves=(3, 10, 21)
            ),
        )
        self.assertEqual(a.loc[0, "u_count_7"], 2)
        self.assertEqual(a.loc[0, "ui_count_all"], 1)
        self.assertEqual(a.loc[1, "u_count_all"], 0)
        self.assertEqual(a.loc[1, "u_recency"], 9999)
        base = manual_columns(a.columns)
        bundles = feature_bundles(a.columns)
        self.assertEqual(set(base) | set(sum(bundles.values(), [])), set(a.columns))
        self.assertFalse(set(base) & set(sum(bundles.values(), [])))

    def test_ranking_known_answer_and_ties(self):
        q = pd.DataFrame(
            dict(snapshot=[0] * 3, user_id=[1] * 3, item_id=[1, 2, 3], label=[1, 0, 1])
        )
        r = ranking(q, [0.3, 0.2, 0.1], k=2).iloc[0]
        self.assertAlmostEqual(r.recall, 0.5)
        self.assertAlmostEqual(r.ndcg, 1 / (1 + 1 / np.log2(3)))
        self.assertAlmostEqual(ranking(q, [1, 1, 1], k=2).iloc[0].ndcg, r.ndcg)


if __name__ == "__main__":
    unittest.main()

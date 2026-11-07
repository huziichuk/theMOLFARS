"""python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline"))
import build_scene as B  # noqa: E402

PROPS = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity"]
P = {p: i for i, p in enumerate(PROPS)}


def rows(n):
    return np.zeros((n, len(PROPS)), np.float32)


class Sanitize(unittest.TestCase):
    def test_clean_file_kept_whole(self):
        self.assertEqual(len(B.sanitize(rows(3000), P)), 3000)

    def test_garbage_tail_is_cut(self):
        # like the broken starter file: valid head, then noise where ~15% of rows look valid by chance
        r = rows(6000)
        tail = np.random.default_rng(0).random(5000) > 0.15
        r[1000:][tail, 0] = np.nan
        self.assertEqual(len(B.sanitize(r, P)), 1000)

    def test_bad_block_mid_file_keeps_good_rows_after_it(self):
        r = rows(4000)
        r[1000:1600, 0] = np.nan
        self.assertEqual(len(B.sanitize(r, P)), 3400)


class PostgresBackend(unittest.TestCase):
    def test_asset_context_from_postgres(self):
        sys.path.insert(0, str(Path(__file__).parents[1]))
        import server  # noqa: E402

        self.addCleanup(server.engine.dispose)
        ctx = server.asset_context("ASSET-017")
        self.assertEqual(ctx["asset"]["tag"], "P-005")
        self.assertEqual(ctx["state"]["issue_code"], "FOULING")
        self.assertIn("process_flow_pct", ctx["telemetry_trend_3d"])
        self.assertTrue(len(ctx["alarms"]) > 0)
        self.assertTrue(len(ctx["maintenance"]) > 0)


if __name__ == "__main__":
    unittest.main()

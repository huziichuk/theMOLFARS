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


class NavGrid(unittest.TestCase):
    """A person (radius 0.22 m) must pass a 0.6 m gap in a wall and must not pass a 0.3 m one."""

    @staticmethod
    def room_with_gap(gap):
        g = np.arange(-1.5, 1.5, 0.03)
        fx, fz = np.meshgrid(g, g)
        floor = np.c_[fx.ravel(), np.zeros(fx.size), fz.ravel()]
        wx, wy = np.meshgrid(np.arange(-1.5, 1.5, 0.02), np.arange(0.3, 1.5, 0.05))
        keep = np.abs(wx.ravel()) > gap / 2
        wall = np.c_[wx.ravel()[keep], wy.ravel()[keep], np.zeros(keep.sum())]
        w = np.r_[floor, wall]
        return B.build_grid(w, np.ones(len(w)), 0.0)

    def both_sides_walkable(self, g):
        zs = (np.nonzero(g["walk"])[1] + 0.5) * B.CELL + g["origin"][1]
        return (zs < -0.4).any() and (zs > 0.4).any()

    def test_wide_gap_connects_both_halves(self):
        self.assertTrue(self.both_sides_walkable(self.room_with_gap(0.6)))

    def test_narrow_gap_does_not(self):
        self.assertFalse(self.both_sides_walkable(self.room_with_gap(0.3)))


class AssetMapping(unittest.TestCase):
    def test_worst_asset_goes_to_the_box_that_fits_never_to_a_wall(self):
        import tempfile

        import pandas as pd

        def box(size, cls, vision_id, seen=True):
            s = np.array(size, float)
            return dict(bmin=-s / 2, bmax=s / 2, cls=cls, vision_id=vision_id, vision_score=0.6, seen=seen)

        wall = box([0.5, 0.5, 0.05], "PROCESS_PUMP", "STRUCTURE")  # pump-sized, but vision says it is a wall
        pump = box([0.4, 0.35, 0.3], "PROCESS_PUMP", "PUMP")
        slab = box([1.5, 1.2, 0.6], "TANK", "TANK")
        with tempfile.TemporaryDirectory() as d:
            xlsx = Path(d) / "pack.xlsx"
            with pd.ExcelWriter(xlsx) as w:
                pd.DataFrame([{"asset_id": "A-1", "tag": "P-1", "asset_class": "PROCESS_PUMP"},
                              {"asset_id": "A-2", "tag": "P-2", "asset_class": "PROCESS_PUMP"}]).to_excel(w, sheet_name="Asset_Master", index=False)
                pd.DataFrame([{"asset_id": "A-1", "health_score": 40, "condition_state": "critical", "issue_code": "X", "urgency": "high"},
                              {"asset_id": "A-2", "health_score": 90, "condition_state": "ok", "issue_code": "", "urgency": "low"}]
                             ).to_excel(w, sheet_name="Current_State", index=False)
            target = B.map_assets([wall, pump, slab], xlsx, np.random.default_rng(0))
        self.assertEqual(target["asset_id"], "A-1")
        self.assertEqual(pump["asset"]["asset_id"], "A-1")  # worst health gets the box that fits
        self.assertNotIn("asset", wall)


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

    def test_cross_site_text_plain_post_is_refused(self):
        # a text/plain POST needs no CORS preflight, so any website could send it to localhost (CSRF)
        import asyncio

        from starlette.requests import Request

        sys.path.insert(0, str(Path(__file__).parents[1]))
        import server  # noqa: E402

        self.addCleanup(server.engine.dispose)

        async def call(ctype):
            req = Request({"type": "http", "method": "POST", "path": "/api/vision", "headers": [(b"content-type", ctype)]})

            async def passed(_):
                return "passed"

            return await server.json_posts_only(req, passed)

        self.assertEqual(asyncio.run(call(b"text/plain")).status_code, 415)
        self.assertEqual(asyncio.run(call(b"application/json")), "passed")

    def test_confidence_drops_with_data_quality_flags_and_weak_identity(self):
        # Builder question Q-06: poor sensor data or a weak 3D mapping must lower the confidence we show
        sys.path.insert(0, str(Path(__file__).parents[1]))
        import server  # noqa: E402

        self.addCleanup(server.engine.dispose)
        clean = server.trust(0.9, [], {"id": "OBJ-1", "vision_score": 0.95})
        flagged = server.trust(0.9, [{"severity": "high"}, {"severity": "medium"}], {"id": "OBJ-1", "vision_score": 0.95})
        unmapped = server.trust(0.9, [], None)
        self.assertEqual(clean["overall"], 0.9)
        self.assertEqual(flagged["after_data_quality"], round(0.9 * 0.85 * 0.93, 2))
        self.assertLess(flagged["overall"], clean["overall"])
        self.assertEqual(unmapped["overall"], 0.0)  # no 3D box: we cannot say which device it is

    def test_object_scene_context_matches_scene(self):
        sys.path.insert(0, str(Path(__file__).parents[1]))
        import server  # noqa: E402

        self.addCleanup(server.engine.dispose)
        obj = SAVED["scene"]["objects"][0]
        sctx = server.object_scene_context(obj["id"])
        self.assertEqual(sctx["object_id"], obj["id"])
        self.assertEqual(sctx["dimensions_m"]["height_y"], obj["size"][1])


def _load_saved():
    import json

    root = Path(__file__).parents[1]
    return dict(
        gt=json.loads((root / "tests" / "vision_eval.json").read_text())["labels"],
        labels=json.loads((root / "web" / "vision_labels.json").read_text()),
        scene=json.loads((root / "web" / "scene.json").read_text()),
    )


SAVED = _load_saved()


class SavedVisionLabels(unittest.TestCase):
    """Checks the committed web/ outputs; no Gemini text is asserted, so re-running gemini_vision.py keeps this green."""

    def test_labels_belong_to_current_scene(self):
        self.assertEqual(SAVED["labels"]["scene_id"], SAVED["scene"]["scene_id"])

    def test_benchmark_on_hand_labelled_crops(self):
        vl = SAVED["labels"]["labels"]
        correct = sum(1 for oid, truth in SAVED["gt"].items() if (vl.get(oid, {}).get("final") or {}).get("id") == truth)
        self.assertGreaterEqual(correct / len(SAVED["gt"]), 0.75)

    def test_every_object_has_vision_fields_and_target_is_equipment(self):
        objs = SAVED["scene"]["objects"]
        for o in objs:
            for k in ("vision_id", "vision_score", "vision_source", "shape_label"):
                self.assertIn(k, o, o["id"])
        mapped = [o for o in objs if o.get("asset")]
        self.assertEqual(len(mapped), 30)
        self.assertTrue(all(o["vision_id"] in ("PUMP", "MOTOR", "VALVE", "GAUGE", "SENSOR") for o in mapped))


if __name__ == "__main__":
    unittest.main()

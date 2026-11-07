"""python -m unittest discover tests"""
import os
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


class InspectionTour(unittest.TestCase):
    """Builder question Q-04: priority from condition, criticality and uncertainty; urgent stops first."""

    def test_priority_rises_with_damage_criticality_and_uncertainty(self):
        base = B.priority(60, "B - important", 0.8)
        self.assertGreater(B.priority(40, "B - important", 0.8), base)   # less healthy
        self.assertGreater(B.priority(60, "A - production critical", 0.8), base)  # more critical
        self.assertGreater(B.priority(60, "B - important", 0.6), base)   # less certain
        self.assertEqual(B.priority(100, "A - production critical", 0.5), 0)

    def test_saved_tour_visits_every_non_healthy_asset_urgent_first(self):
        t = SAVED["scene"]["tour"]
        sick = {o["asset"]["asset_id"] for o in SAVED["scene"]["objects"] if o.get("asset") and o["asset"]["condition_state"] != "healthy"}
        self.assertEqual({st["asset_id"] for st in t["stops"]} | set(t["skipped"]) >= sick, True)
        tiers = [B.URGENCY_ORDER[st["urgency"]] for st in t["stops"]]
        self.assertEqual(tiers, sorted(tiers))
        idx = [st["path_index"] for st in t["stops"]]
        self.assertEqual(idx, sorted(idx))
        self.assertLess(idx[-1], len(t["path"]))


class FetchData(unittest.TestCase):
    """The splat download keeps only complete .ply files and refuses an HTML error page."""

    def test_downloads_a_ply_and_rejects_anything_else(self):
        import tempfile

        import fetch_data

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "src.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n" + bytes(1000))
            (d / "page.html").write_bytes(b"<html>quota exceeded</html>")
            got = fetch_data.fetch_ply(d / "out" / "room.ply", (d / "src.ply").as_uri())
            self.assertEqual(got.read_bytes(), (d / "src.ply").read_bytes())
            with self.assertRaises(SystemExit):
                fetch_data.fetch_ply(d / "bad.ply", (d / "page.html").as_uri())
            self.assertFalse((d / "bad.ply").exists())


class UsdExport(unittest.TestCase):
    """Q-09: the committed scene exports to OpenUSD; with usd-core installed (CI) the stage is opened for real."""

    def test_every_object_becomes_a_cube_with_its_asset(self):
        import export_usd

        usda = export_usd.scene_to_usda(SAVED["scene"])
        self.assertTrue(usda.startswith("#usda 1.0"))
        self.assertEqual(usda.count('def Cube "OBJ_'), len(SAVED["scene"]["objects"]))
        self.assertIn('valmet:tag = "P-005"', usda)
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            return
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "s.usda"
            f.write_text(usda, encoding="utf-8")
            stage = Usd.Stage.Open(str(f))
            self.assertEqual(UsdGeom.GetStageUpAxis(stage), "Y")
            target = stage.GetPrimAtPath("/BoilerRoom/Objects/" + SAVED["scene"]["target_object"].replace("-", "_"))
            self.assertEqual(target.GetAttribute("valmet:tag").Get(), "P-005")


class Frames(unittest.TestCase):
    """Item 6: oriented frames for rotated objects and one group for touching parts of the same thing."""

    def test_rotated_plank_gets_an_oriented_box_axis_aligned_one_does_not(self):
        rng = np.random.default_rng(1)
        u, v = rng.uniform(-0.5, 0.5, 4000), rng.uniform(-0.05, 0.05, 4000)
        for yaw, want in ((np.pi / 4, True), (0.0, False)):
            x, z = u * np.cos(yaw) - v * np.sin(yaw), u * np.sin(yaw) + v * np.cos(yaw)
            q = np.c_[x, rng.uniform(0, 0.3, 4000), z]
            box = B.oriented_box(q, np.percentile(q, 2, 0), np.percentile(q, 98, 0))
            self.assertEqual(box is not None, want)
            if box:
                self.assertAlmostEqual(abs(np.sin(2 * box["yaw"])), 1.0, places=2)  # +-45 deg, either axis sign
                self.assertAlmostEqual(max(box["size"][0], box["size"][2]), 0.96, delta=0.03)

    def test_touching_parts_of_one_class_group_assets_never_do(self):
        def box(x0, vid, asset=None):
            o = dict(bmin=np.array([x0, 0, 0.0]), bmax=np.array([x0 + 1, 1, 1.0]), cls=vid, vision_id=vid)
            if asset:
                o["asset"] = asset
            return o
        objs = [box(0, "PIPE"), box(1.02, "PIPE"), box(2.04, "PIPE", {"asset_id": "A"}), box(5, "PIPE"), box(0, "BOX")]
        groups = B.group_fragments(objs)
        self.assertEqual([g["members"] for g in groups], [["OBJ-000", "OBJ-001"]])
        self.assertNotIn("group", objs[2])


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
                pd.DataFrame([{"asset_id": "A-1", "health_score": 40, "condition_state": "critical", "issue_code": "X", "urgency": "high", "model_confidence": 0.9},
                              {"asset_id": "A-2", "health_score": 90, "condition_state": "ok", "issue_code": "", "urgency": "low", "model_confidence": 0.9}]
                             ).to_excel(w, sheet_name="Current_State", index=False)
            target = B.map_assets([wall, pump, slab], xlsx, np.random.default_rng(0))
        self.assertEqual(target["asset_id"], "A-1")
        self.assertEqual(pump["asset"]["asset_id"], "A-1")  # worst health gets the box that fits
        self.assertNotIn("asset", wall)


@unittest.skipUnless(os.environ.get("DATABASE_URL") or (Path(__file__).parents[1] / "data" / "builder_pack.xlsx").exists(),
                     "needs a database or data/builder_pack.xlsx (not in git)")
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

    def test_citations_are_checked_against_the_evidence(self):
        # item 3: an ID the model invents must be flagged; everything the templates cite must be found
        sys.path.insert(0, str(Path(__file__).parents[1]))
        import server  # noqa: E402

        self.addCleanup(server.engine.dispose)
        ctx = server.asset_context("ASSET-017")
        real = ctx["alarms"][-1]["event_id"]
        c = server.check_citations(f"Pump fouling [{real}, EVT-99999] see [DOC-NOT-REAL]", ctx)
        self.assertEqual(c["verified"], [real])
        self.assertEqual(c["unknown"], ["DOC-NOT-REAL", "EVT-99999"])
        report = server.offline_report(ctx, "ARRIVED")
        self.assertEqual(server.check_citations(report, ctx)["unknown"], [])
        line = server.brief_line(ctx)  # item 2: the line said while walking past
        self.assertIn("P-005", line)
        self.assertEqual(server.check_citations(line, ctx)["unknown"], [])
        self.assertTrue(server.check_citations(line, ctx)["verified"])

    def test_telemetry_detector_flags_a_late_step_not_noise(self):
        # Q-02: a signal that steps up in the last 6 h is anomalous, a noisy but stable one is not
        import pandas as pd

        sys.path.insert(0, str(Path(__file__).parents[1]))
        import server  # noqa: E402

        self.addCleanup(server.engine.dispose)
        rng = np.random.default_rng(0)
        n = 144
        stable = 50 + rng.normal(0, 1, n)
        step = 2 + rng.normal(0, 0.1, n)
        step[-12:] += 1.0  # +10 baseline sigmas in the last 6 h
        tel = pd.DataFrame({"timestamp": pd.date_range("2026-10-05", periods=n, freq="30min"), "stable": stable, "vibration": step})
        an = server.detect_anomalies(tel, series=True)
        flags = {x["name"]: x["anomalous"] for x in an["signals"]}
        self.assertEqual(flags, {"vibration": True, "stable": False})
        self.assertEqual(an["signals"][0]["name"], "vibration")  # strongest first
        self.assertEqual(len(an["signals"][0]["series"]), n // 2)

    def test_live_telemetry_reaches_the_detector(self):
        # Q-07: a reading posted like an MQTT/OPC UA bridge would is stored in the same table the detector reads
        import asyncio
        import json

        import pandas as pd
        from sqlalchemy import text
        from starlette.requests import Request

        sys.path.insert(0, str(Path(__file__).parents[1]))
        import server  # noqa: E402

        def cleanup():
            with server.engine.begin() as conn:
                conn.execute(text("DELETE FROM telemetry_sample_3d WHERE source_profile = 'unit-test'"))
            server.engine.dispose()

        self.addCleanup(cleanup)

        async def post(body):
            raw = json.dumps(body).encode()

            async def receive():
                return {"type": "http.request", "body": raw, "more_body": False}

            req = Request({"type": "http", "method": "POST", "path": "/api/telemetry", "headers": [(b"content-type", b"application/json")]}, receive)
            return await server.ingest_telemetry(req)

        ok = asyncio.run(post({"asset_id": "ASSET-029", "timestamp": "2026-10-08T00:00:00", "values": {"vibration_mm_s": 9.9}, "source": "unit-test"}))
        self.assertEqual(ok["stored"], 1)
        bad = asyncio.run(post({"asset_id": "ASSET-029", "values": {"vibration_mm_s; DROP TABLE x": 1}, "source": "unit-test"}))
        self.assertEqual(bad.status_code, 400)
        with server.engine.connect() as conn:
            tel = pd.read_sql(text("SELECT * FROM telemetry_sample_3d WHERE asset_id = 'ASSET-029' ORDER BY timestamp"), conn)
        self.assertEqual(tel["vibration_mm_s"].iloc[-1], 9.9)


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

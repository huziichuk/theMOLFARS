"""Multi-view classification of every 3D box on a GPU PC: several splat renders per box + the matched video tile,
scored by a large vision-language model (SigLIP 2 / CLIP L-14), probabilities averaged over the views.

Unlike video_lift.py this needs no COLMAP alignment: the views are rendered from the splat itself, from the aisle
cells a technician can stand on, so every box gets 3-6 sharp-ish viewpoints instead of the single browser render.

Usage (GPU PC, .venv-gpu):
  python pipeline/multiview_classify.py work/mv2 --render              # 1) zoomed views of every box (cached)
  python pipeline/multiview_classify.py work/mv --render --wide        #    wide context views for vlm_classify.py
  python pipeline/multiview_classify.py work/mv2 --eval                # 2) CLIP-L/14, SigLIP2 on tests/vision_eval.json
  python pipeline/multiview_classify.py work/mv2 --label --model google/siglip2-base-patch16-384
                                                                       # 3) work/mv2/multiview_labels.json (SigLIP vote)
  then pipeline/vlm_classify.py for the VLM label and the merge into web/vision_labels.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import build_scene as B  # noqa: E402

VIEW_PX = 320
VIEW_FOV = 64.0                        # widest field of view (big boxes); narrower = zoom for small boxes
MIN_FOV = 22.0
AZIMUTHS = (0, -45, 45, -90, 90)       # degrees around the nearest-aisle direction
TOP_VIEW = True                        # one extra view from above the aisle
MIN_COVERAGE = 0.5                     # drop renders where the splats cover < 50% (camera outside the room)
MIN_CONTRAST = 12.0                    # ... or the image is one flat blob (camera inside a wall)
MAX_CAM_TO_AISLE = 0.55                # metres: camera must be near a walkable cell
VIDEO_WEIGHT = 2.0                     # the sharp video tile counts as this many renders in the vote
DEFAULT_MODELS = ["openai/clip-vit-large-patch14", "google/siglip2-base-patch16-384", "google/siglip2-so400m-patch14-384"]


# ------------------------------------------------------------------ rendering (pipeline/splat_render.py)
def walkable_points(scene):
    g = scene["grid"]
    pts = [[g["origin"][0] + (i + 0.5) * g["cell"], g["origin"][1] + (j + 0.5) * g["cell"]]
           for j in range(g["h"]) for i in range(g["w"]) if g["walkable"][j * g["w"] + i] == "1"]
    return np.array(pts, np.float64)


def view_cameras(obj, walk_pts):
    """Camera positions around a box: nearest aisle direction +- azimuths, all kept near walkable cells."""
    c, s = np.array(obj["center"], np.float64), np.array(obj["size"], np.float64)
    near = walk_pts[np.argmin(((walk_pts - c[[0, 2]]) ** 2).sum(1))]
    d0 = np.array([near[0] - c[0], near[1] - c[2]])
    d0 = d0 / np.linalg.norm(d0) if np.linalg.norm(d0) > 1e-6 else np.array([0.0, 1.0])
    max_s = max(float(s.max()), 0.25)
    dist = max(0.85, min(2.4, max_s * 1.45))
    # zoom so the box + 80% context fills the frame (the browser crops its render the same way); tiny boxes are
    # capped at MIN_FOV because splats blur below ~10 cm anyway
    extent = max(float(s.max()), 0.10) * 1.8
    fov = float(np.clip(np.degrees(2 * np.arctan(extent / 2 / dist)), MIN_FOV, VIEW_FOV))
    cams = []
    for az in AZIMUTHS:
        a = np.deg2rad(az)
        d = np.array([d0[0] * np.cos(a) - d0[1] * np.sin(a), d0[0] * np.sin(a) + d0[1] * np.cos(a)])
        pos = np.array([c[0] + d[0] * dist, np.clip(c[1] + 0.08, -0.7, 0.6), c[2] + d[1] * dist])
        gap = np.sqrt(((walk_pts - pos[[0, 2]]) ** 2).sum(1).min())
        if az == 0 or gap <= MAX_CAM_TO_AISLE:
            cams.append((f"az{az:+d}", pos))
    if TOP_VIEW:
        pos = np.array([c[0] + d0[0] * dist * 0.8, min(c[1] + 0.45 * dist, 0.75), c[2] + d0[1] * dist * 0.8])
        cams.append(("top", pos))
    return cams, c, fov


def project_box(obj, cam_pos, fov_deg, size=VIEW_PX):
    """2D bbox (x0,y0,x1,y1) of the 3D box in a render made by splat_render.SplatRenderer.render(cam_pos, center)."""
    c, h = np.array(obj["center"], np.float64), np.array(obj["size"], np.float64) / 2
    corners = np.array([c + h * np.array([x, y, z]) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    cam = np.asarray(cam_pos, np.float64)
    fwd = c - cam
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, [0, 1, 0])
    right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-4 else np.array([1.0, 0, 0])
    up = np.cross(right, fwd)
    t = (corners - cam) @ np.stack([right, up, fwd]).T
    z = np.clip(t[:, 2], 0.05, None)
    f = (size / 2) / np.tan(np.deg2rad(fov_deg / 2))
    u, v = f * t[:, 0] / z + size / 2, -f * t[:, 1] / z + size / 2
    return (float(np.clip(u.min(), 0, size)), float(np.clip(v.min(), 0, size)),
            float(np.clip(u.max(), 0, size)), float(np.clip(v.max(), 0, size)))


def view_boxes(obj, walk_pts, files, fov=None):
    """Per kept view file: the projected 2D bbox of the object (views are matched to cameras by name).
    fov overrides the adaptive one for renders made with a fixed field of view (the wide views)."""
    cams, c, fov_auto = view_cameras(obj, walk_pts)
    fov = fov or fov_auto
    by_name = {name: pos for name, pos in cams}
    out = {}
    for fn in files:
        name = fn[len(obj["id"]) + 1:-4]
        if name in by_name:
            out[fn] = project_box(obj, by_name[name], fov)
    return out


def render_all(work, scene, ply, wide=False):
    views_dir = work / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    index_path = work / "views.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    todo = [o for o in scene["objects"] if o["id"] not in index]
    if not todo:
        print(f"[render] all {len(index)} boxes already rendered in {views_dir}")
        return index
    from splat_render import SplatRenderer
    print(f"[render] loading {ply.name} ...")
    R = SplatRenderer(ply)
    walk = walkable_points(scene)
    print(f"[render] {R.n:,} gaussians on {R.dev}, {len(walk)} walkable cells, {len(todo)} boxes to render")
    for n, o in enumerate(todo):
        cams, c, fov = view_cameras(o, walk)
        if wide:
            fov = VIEW_FOV  # context views for the VLM: no zoom, the box is drawn on them later
        kept = []
        for name, pos in cams:
            img, cov = R.render(pos, c, size=VIEW_PX, fov_deg=fov)
            if name != "az+0" and (cov < MIN_COVERAGE or img.std() < MIN_CONTRAST):
                continue  # camera inside a wall: nothing or a flat blob in view
            fn = f"{o['id']}_{name}.jpg"
            Image.fromarray(img).save(views_dir / fn, quality=92)
            kept.append(fn)
        index[o["id"]] = kept
        if (n + 1) % 25 == 0 or n + 1 == len(todo):
            index_path.write_text(json.dumps(index))
            print(f"[render] {n + 1}/{len(todo)}")
    return index


def video_tile(entry, frames_dir, frame_files, min_csls=0.025):
    """The matched sharp tile from the walk-through video (csls from the browser photo pass), or None."""
    p = (entry or {}).get("photo") or {}
    if p.get("csls", 0) < min_csls or p.get("f") is None or not (0 <= int(p["f"]) < len(frame_files)):
        return None
    fp = frames_dir / frame_files[int(p["f"])]
    if not fp.exists():
        return None
    im = Image.open(fp).convert("RGB")
    s = p.get("s", 224)
    pad = int(s * 0.2)
    x0, y0 = int(p.get("x", 0)), int(p.get("y", 0))
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x0 + s + pad), min(im.height, y0 + s + pad)))


# ------------------------------------------------------------------ models
class Scorer:
    """Zero-shot scorer: one text embedding per class = mean over prompts x templates (same recipe as the browser)."""

    def __init__(self, name, vocab, dev="cuda"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.dev, self.name = torch, dev, name
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
        self.proc = AutoProcessor.from_pretrained(name)
        self.siglip = "siglip" in name.lower()
        feat = lambda o: o if torch.is_tensor(o) else o.pooler_output  # transformers >= 5 returns ModelOutput
        self.feat = feat
        embs = []
        with torch.no_grad():
            for c in vocab["classes"]:
                texts = [t.format(p) for p in c["prompts"] for t in vocab["templates"]]
                kw = dict(padding="max_length", max_length=64, truncation=True) if self.siglip else dict(padding=True)
                t = self.proc(text=texts, return_tensors="pt", **kw).to(dev)
                e = torch.nn.functional.normalize(feat(self.model.get_text_features(**t)).float(), dim=-1).mean(0)
                embs.append(e)
        self.text = torch.nn.functional.normalize(torch.stack(embs), dim=-1)
        self.scale = float(self.model.logit_scale.exp()) if hasattr(self.model, "logit_scale") else 100.0

    def probs(self, images):
        torch = self.torch
        with torch.no_grad():
            x = self.proc(images=images, return_tensors="pt").to(self.dev)
            x = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in x.items()}
            im = torch.nn.functional.normalize(self.feat(self.model.get_image_features(**x)).float(), dim=-1)
            return (self.scale * im @ self.text.T).softmax(-1).cpu().numpy()


def score_objects(scorer, scene, index, work, vl, frames_dir, frame_files, ids=None):
    """Per object: probabilities for (a) nearest-aisle render only, (b) all renders, (c) renders + video tile."""
    out = {}
    for o in scene["objects"]:
        oid = o["id"]
        if ids is not None and oid not in ids:
            continue
        files = index.get(oid, [])
        if not files:
            continue
        ims = [Image.open(work / "views" / f).convert("RGB") for f in files]
        p = scorer.probs(ims)
        tile = video_tile(vl["labels"].get(oid), frames_dir, frame_files)
        pv = scorer.probs([tile])[0] if tile is not None else None
        allv = p.mean(0)
        fused = (p.sum(0) + VIDEO_WEIGHT * pv) / (len(p) + VIDEO_WEIGHT) if pv is not None else allv
        out[oid] = dict(single=p[0], multi=allv, fused=fused, n_views=len(p), video=pv is not None)
    return out


def evaluate(results, gt, classes):
    ids = [c["id"] for c in classes]
    rep = {}
    for key in ("single", "multi", "fused"):
        correct = sum(1 for oid, t in gt.items() if oid in results and ids[int(results[oid][key].argmax())] == t)
        rep[key] = correct
    return rep


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--wide", action="store_true", help="with --render: fixed wide field of view (context views for vlm_classify.py)")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--model", default=DEFAULT_MODELS[-1])
    ap.add_argument("--apply", type=Path, help="merge work/multiview_labels.json into this vision_labels.json")
    ap.add_argument("--scene", type=Path, default=ROOT / "web" / "scene.json")
    ap.add_argument("--labels", type=Path, default=ROOT / "web" / "vision_labels.json")
    ap.add_argument("--ply", type=Path, default=ROOT / "data" / "equipment_room.ply")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)

    scene = json.loads(a.scene.read_text())
    vocab = json.loads((ROOT / "web" / "vocab.json").read_text())
    vl = json.loads(a.labels.read_text())
    frames_dir = ROOT / "web" / "frames"
    frame_files = json.loads((frames_dir / "index.json").read_text())["files"] if (frames_dir / "index.json").exists() else []
    index = render_all(a.work, scene, a.ply, wide=a.wide) if a.render else json.loads((a.work / "views.json").read_text())

    if a.eval:
        gt = json.loads((ROOT / "tests" / "vision_eval.json").read_text())["labels"]
        rows = []
        for name in a.models:
            try:
                sc = Scorer(name, vocab)
            except Exception as e:  # gated / not downloadable: report and continue
                print(f"[eval] {name}: skipped ({type(e).__name__}: {str(e)[:120]})")
                continue
            res = score_objects(sc, scene, index, a.work, vl, frames_dir, frame_files, ids=set(gt))
            rep = evaluate(res, gt, vocab["classes"])
            rows.append((name, rep))
            print(f"[eval] {name}: single-view {rep['single']}/50  multi-view {rep['multi']}/50  +video {rep['fused']}/50")
            del sc
        (a.work / "eval.json").write_text(json.dumps(rows, indent=1))

    if a.label:
        sc = Scorer(a.model, vocab)
        res = score_objects(sc, scene, index, a.work, vl, frames_dir, frame_files)
        ids = vocab["classes"]
        labels = {}
        for oid, r in res.items():
            k = int(r["fused"].argmax())
            c = ids[k]
            labels[oid] = dict(id=c["id"], label=c["name"], kind=c["kind"], asset_class=c["asset_class"],
                               score=round(float(r["fused"][k]), 3), source="multiview", model=a.model,
                               views=int(r["n_views"]) + (1 if r["video"] else 0), video=bool(r["video"]),
                               top3=[(ids[i]["id"], round(float(r["fused"][i]), 3)) for i in np.argsort(-r["fused"])[:3]])
        outp = a.work / "multiview_labels.json"
        outp.write_text(json.dumps(dict(scene_id=scene["scene_id"], model=a.model, labels=labels), indent=0))
        conf = sum(v["score"] >= vocab["min_score"] for v in labels.values())
        print(f"[label] {len(labels)} boxes, {conf} with score >= {vocab['min_score']}, {sum(v['video'] for v in labels.values())} used a video tile -> {outp}")

    if a.apply:
        mv = json.loads((a.work / "multiview_labels.json").read_text())
        target = json.loads(a.apply.read_text())
        if mv["scene_id"] != target["scene_id"]:
            sys.exit(f"scene_id mismatch: {mv['scene_id']} vs {target['scene_id']}")
        n = 0
        for oid, lab in mv["labels"].items():
            target["labels"].setdefault(oid, {})["multiview"] = lab
            n += 1
        tmp = a.apply.with_suffix(".tmp")
        tmp.write_text(json.dumps(target))
        tmp.replace(a.apply)
        print(f"[apply] multiview entry added to {n} boxes in {a.apply}")


if __name__ == "__main__":
    main()

"""Option 2: lift the walk-through video into the splat's 3D frame and relabel every 3D box from sharp multi-view photos.

Run on an x64 machine with a GPU (CUDA PyTorch). Steps:
  1. sample frames (ffmpeg)                                   -> work/images/
  2. SfM with pycolmap (SIFT + sequential matching + mapping) -> work/sparse/
  3. align COLMAP frame to the splat frame (scale+rigid): extent-based scale, FPFH+RANSAC, ICP (Open3D)
     or, if that fails, from >=4 hand-picked point pairs (--pairs pairs.json)
  4. project each 3D box from web/scene.json into every registered frame, crop the best views,
     CLIP-classify each crop and average the probabilities over views (multi-view vote)
  5. OWLv2 open-vocabulary detection on frames; lift each detection to 3D from the COLMAP points
     that fall inside it, keep the ones that don't overlap an existing box -> new objects
Writes web/video_labels.json, which the viewer applies on load.

Usage:
  python pipeline/video_lift.py data/boiler_room_360.mp4 web/scene.json data/equipment_room.ply work --fps 2
  python pipeline/video_lift.py --selftest           # math checks, no GPU / COLMAP needed
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# same vocabulary as the browser pass (web/index.html VOCAB)
VOCAB = [
    ("red circulation pump mounted on a pipe", "PROCESS_PUMP"), ("centrifugal pump with electric motor", "PROCESS_PUMP"),
    ("electric motor", "ELECTRIC_MOTOR"), ("control valve with actuator", "CONTROL_VALVE"), ("hand wheel valve", "CONTROL_VALVE"),
    ("pressure gauge", "PRESSURE_TRANSMITTER"), ("electrical control cabinet", "CABINET"), ("red expansion vessel tank", "TANK"),
    ("blue pressure tank", "TANK"), ("white insulated pipe", "PIPE"), ("ladder", "CLUTTER"), ("red plastic storage box", "CLUTTER"),
    ("cardboard box", "CLUTTER"), ("wire mesh cart", "CLUTTER"), ("office chair", "CLUTTER"), ("bare concrete wall", "CLUTTER"),
]
EQUIPMENT = [p for p, c in VOCAB if c not in ("CLUTTER", "PIPE")]
PROMPT = "a photo of a {} in an industrial boiler room"
MAX_VIEWS = 6          # best views per box used in the vote
MIN_BOX_PX = 40        # ignore projections smaller than this


# ------------------------------------------------------------------ math (self-tested)
def umeyama(src, dst):
    """Similarity transform dst ~ s*R@src + t from paired points (N>=3)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / a.var(0).sum()
    return s, R, md - s * R @ ms


def to_matrix(s, R, t):
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def box_corners(center, size):
    c, h = np.asarray(center), np.asarray(size) / 2
    return np.array([c + h * np.array([x, y, z]) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def project(P_cam_from_world, K, pts, wh):
    """Pinhole projection; returns pixel bbox (x0,y0,x1,y1) or None if behind camera / off-image / tiny."""
    X = (P_cam_from_world[:3, :3] @ pts.T + P_cam_from_world[:3, 3:4])
    if (X[2] <= 0.05).any():
        return None
    uv = (K @ (X / X[2]))[:2].T
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    W, H = wh
    if x1 < 0 or y1 < 0 or x0 > W or y0 > H:
        return None
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if min(x1 - x0, y1 - y0) < MIN_BOX_PX:
        return None
    return float(x0), float(y0), float(x1), float(y1), float(np.median(X[2]))


def selftest():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    ang = 0.7
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    dst = 2.5 * src @ R.T + np.array([1, -2, 3])
    s, R2, t = umeyama(src, dst)
    assert abs(s - 2.5) < 1e-6 and np.allclose(R2, R) and np.allclose(t, [1, -2, 3]), "umeyama"
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    P = np.hstack([np.eye(3), [[0], [0], [0]]])
    bb = project(P, K, box_corners([0, 0, 3], [0.6, 0.6, 0.6]), (640, 480))
    assert bb and abs((bb[0] + bb[2]) / 2 - 320) < 1 and bb[2] - bb[0] > MIN_BOX_PX, "project in view"
    assert project(P, K, box_corners([0, 0, -3], [0.6] * 3), (640, 480)) is None, "behind camera"
    print("selftest ok")


# ------------------------------------------------------------------ 1-2: frames + SfM
def sample_frames(video, img_dir, fps):
    img_dir.mkdir(parents=True, exist_ok=True)
    if not any(img_dir.glob("*.jpg")):
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2",
                        str(img_dir / "%05d.jpg")], check=True)
    print(f"[frames] {len(list(img_dir.glob('*.jpg')))} frames")


def run_sfm(img_dir, work):
    import pycolmap
    sparse = work / "sparse"
    if (sparse / "0").exists():
        return pycolmap.Reconstruction(str(sparse / "0"))
    db = work / "database.db"
    pycolmap.extract_features(str(db), str(img_dir), camera_mode=pycolmap.CameraMode.SINGLE)
    pycolmap.match_sequential(str(db))
    sparse.mkdir(exist_ok=True)
    recs = pycolmap.incremental_mapping(str(db), str(img_dir), str(sparse))
    if not recs:
        sys.exit("COLMAP registered no images (motion blur / too few features). Try a higher --fps.")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    print(f"[sfm] {best.num_reg_images()} images registered, {best.num_points3D()} points")
    return best


def cam_from_world(img):
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world  # API differs across versions
    return np.asarray(cfw.matrix())


# ------------------------------------------------------------------ 3: alignment
def splat_points(ply):
    raw = Path(ply).read_bytes()
    end = raw.index(b"end_header\n") + 11
    props = [l.split()[-1] for l in raw[:end].decode().splitlines() if l.startswith("property")]
    rows = np.frombuffer(raw[end:], dtype="<f4").reshape(-1, len(props))
    P = {p: i for i, p in enumerate(props)}
    op = 1 / (1 + np.exp(-rows[:, P["opacity"]].astype(np.float64)))
    xyz = rows[op > 0.5, :3].astype(np.float64)
    xyz = xyz[np.isfinite(xyz).all(1) & (np.abs(xyz) < 50).all(1)]
    return np.stack([xyz[:, 0], -xyz[:, 1], -xyz[:, 2]], 1)  # same world frame as build_scene.to_world


def align(rec, splat_xyz, pairs_file=None):
    """Returns 4x4 matrix world_from_colmap."""
    if pairs_file:
        pairs = json.loads(Path(pairs_file).read_text())  # [{"colmap":[x,y,z],"world":[x,y,z]}, ...]
        s, R, t = umeyama(np.array([p["colmap"] for p in pairs]), np.array([p["world"] for p in pairs]))
        print(f"[align] from {len(pairs)} hand-picked pairs, scale {s:.3f}")
        return to_matrix(s, R, t)
    import open3d as o3d
    reg = o3d.pipelines.registration
    col = np.array([p.xyz for p in rec.points3D.values()])
    col = col[np.linalg.norm(col - np.median(col, 0), axis=1) < np.percentile(np.linalg.norm(col - np.median(col, 0), axis=1), 95)]
    ext = lambda a: np.percentile(a, 95, 0) - np.percentile(a, 5, 0)
    s0 = np.median(ext(splat_xyz)) / np.median(ext(col))  # rough scale so FPFH radii match
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((col - col.mean(0)) * s0))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(splat_xyz))
    v = 0.05
    src_d, tgt_d = src.voxel_down_sample(v), tgt.voxel_down_sample(v)
    for pc in (src_d, tgt_d):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2, max_nn=30))
    f = lambda pc: reg.compute_fpfh_feature(pc, o3d.geometry.KDTreeSearchParamHybrid(radius=v * 5, max_nn=100))
    ransac = reg.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, f(src_d), f(tgt_d), True, v * 1.5, reg.TransformationEstimationPointToPoint(True), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(v * 1.5)],
        reg.RANSACConvergenceCriteria(400000, 0.999))
    icp = reg.registration_icp(src_d, tgt_d, v * 1.0, ransac.transformation, reg.TransformationEstimationPointToPoint(True))
    print(f"[align] RANSAC fitness {ransac.fitness:.2f} -> ICP fitness {icp.fitness:.2f}, rmse {icp.inlier_rmse:.3f}")
    if icp.fitness < 0.3:
        print("[align] WARNING: low fitness; check work/align_check.ply or pass --pairs with 4+ hand-picked points")
    pre = to_matrix(s0, np.eye(3), -col.mean(0) * s0)
    return np.asarray(icp.transformation) @ pre


# ------------------------------------------------------------------ 4-5: multi-view labels + new objects
def main(a):
    work = Path(a.work)
    img_dir = work / "images"
    sample_frames(a.video, img_dir, a.fps)
    rec = run_sfm(img_dir, work)
    W_from_C = align(rec, splat_points(a.ply), a.pairs)
    C_from_W = np.linalg.inv(W_from_C)
    scale = np.cbrt(abs(np.linalg.det(W_from_C[:3, :3])))
    scene = json.loads(Path(a.scene).read_text())

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[models] torch on {dev}")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()  # bigger than the browser model
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    with torch.no_grad():
        t = cproc(text=[PROMPT.format(p) for p, _ in VOCAB], return_tensors="pt", padding=True).to(dev)
        text_emb = torch.nn.functional.normalize(clip.get_text_features(**t), dim=-1)

    def clip_probs(crops):
        with torch.no_grad():
            x = cproc(images=crops, return_tensors="pt").to(dev)
            im = torch.nn.functional.normalize(clip.get_image_features(**x), dim=-1)
            return (100 * im @ text_emb.T).softmax(-1).cpu().numpy()

    # camera list in world-aligned form: P maps splat-world -> camera
    cams = []
    for img in rec.images.values():
        cam = rec.cameras[img.camera_id]
        K = np.asarray(cam.calibration_matrix())
        P = cam_from_world(img) @ C_from_W  # 3x4 @ 4x4
        cams.append((img, K, P, (cam.width, cam.height)))

    labels = {}
    for o in scene["objects"]:
        corners = np.c_[box_corners(o["center"], o["size"]), np.ones(8)]
        views = []
        for img, K, P, wh in cams:
            bb = project(P, K, corners[:, :3], wh)
            if bb:
                views.append(((bb[2] - bb[0]) * (bb[3] - bb[1]) / bb[4], img, bb))  # big + close first
        views.sort(key=lambda v: -v[0])
        crops = []
        for _, img, (x0, y0, x1, y1, _) in views[:MAX_VIEWS]:
            pad = 0.4 * max(x1 - x0, y1 - y0)
            im = Image.open(img_dir / img.name).convert("RGB")
            crops.append(im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad))))
        if not crops:
            continue
        p = clip_probs(crops).mean(0)  # multi-view vote
        k = int(p.argmax())
        labels[o["id"]] = dict(label=VOCAB[k][0], cls=VOCAB[k][1], score=round(float(p[k]), 3), views=len(crops),
                               frames=[img.name for _, img, _ in views[:MAX_VIEWS]])
    print(f"[labels] {len(labels)}/{len(scene['objects'])} boxes seen in video, "
          f"{sum(v['score'] >= 0.3 for v in labels.values())} confident")

    # 5) OWLv2 on frames -> lift detections with COLMAP points inside each 2D box
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    oproc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    pts3d = {pid: np.append(p.xyz, 1) for pid, p in rec.points3D.items()}
    found = []
    for img, K, P, wh in cams[:: max(1, len(cams) // 120)]:  # ~120 frames is plenty
        im = Image.open(img_dir / img.name).convert("RGB")
        with torch.no_grad():
            x = oproc(text=[EQUIPMENT], images=im, return_tensors="pt").to(dev)
            out = owl(**x)
        sz = max(im.size)
        det = oproc.post_process_object_detection(out, threshold=0.25, target_sizes=torch.tensor([[sz, sz]]).to(dev))[0]
        obs = [(pt.xy, pt.point3D_id) for pt in img.points2D if pt.has_point3D()]
        for box, score, lab in zip(det["boxes"].cpu().numpy(), det["scores"].cpu().numpy(), det["labels"].cpu().numpy()):
            inside = [W_from_C @ pts3d[pid] for xy, pid in obs if box[0] <= xy[0] <= box[2] and box[1] <= xy[1] <= box[3]]
            if len(inside) < 8:
                continue
            q = np.array(inside)[:, :3]
            med = np.median(q, 0)
            q = q[np.linalg.norm(q - med, axis=1) < 3 * np.median(np.linalg.norm(q - med, axis=1)) + 1e-6]
            lo, hi = np.percentile(q, 10, 0), np.percentile(q, 90, 0)
            if (hi - lo).max() > 2.5 or (hi - lo).max() < 0.05:
                continue
            found.append(dict(label=EQUIPMENT[int(lab)], score=round(float(score), 3), bmin=lo, bmax=hi))

    def overlap(a_lo, a_hi, b_lo, b_hi):
        inter = np.clip(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0, None).prod()
        return inter / max(min((a_hi - a_lo).prod(), (b_hi - b_lo).prod()), 1e-9)

    existing = [(np.array(o["center"]) - np.array(o["size"]) / 2, np.array(o["center"]) + np.array(o["size"]) / 2)
                for o in scene["objects"]]
    extra = []
    for d in sorted(found, key=lambda d: -d["score"]):
        if all(overlap(d["bmin"], d["bmax"], lo, hi) < 0.3 for lo, hi in existing):
            existing.append((d["bmin"], d["bmax"]))
            extra.append(dict(id=f"VID-{len(extra):03d}", label=d["label"], score=d["score"],
                              cls=dict(VOCAB)[d["label"]], center=((d["bmin"] + d["bmax"]) / 2).round(3).tolist(),
                              size=(d["bmax"] - d["bmin"]).round(3).tolist()))
    print(f"[owlv2] {len(found)} detections lifted to 3D -> {len(extra)} new objects")

    out = Path(a.scene).with_name("video_labels.json")
    out.write_text(json.dumps(dict(world_from_colmap=W_from_C.round(6).tolist(), colmap_scale=float(scale),
                                   registered_frames=len(cams), labels=labels, extra_objects=extra)))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("scene"); ap.add_argument("ply"); ap.add_argument("work")
    ap.add_argument("--fps", default="2")
    ap.add_argument("--pairs", help="JSON list of {colmap:[x,y,z], world:[x,y,z]} (>=4) if auto alignment fails")
    main(ap.parse_args())

"""Offline pipeline: raw 3DGS .ply -> clean splat + semantic objects + nav grid + A* route -> web/scene.json.

Usage: python pipeline/build_scene.py data/equipment_room.ply data/builder_pack.xlsx web
         [--vision web/vision_labels.json]   map Excel assets by the classes the browser vision pass saw
         [--boxes DIR]                        add non-overlapping boxes from the ml branch's segment.py (objects.json + transform.json)
"""
import argparse
import hashlib
import json
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

VOXEL = 0.04          # m, object segmentation voxel
COLOR_TOL = 0.10      # max per-channel colour gap between joined voxels
MIN_PTS = 30          # splats per object
CELL = 0.10           # m, nav grid cell
ROBOT_R = 0.22        # m, avatar radius for obstacle inflation
HEAD = 1.7            # m, obstacles counted between floor+0.2 and floor+HEAD


# ---------------------------------------------------------------- 1. load + sanitize
def load_ply(path):
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode()
    props = [l.split()[-1] for l in header.splitlines() if l.startswith("property")]
    assert all(l.split()[1] == "float" for l in header.splitlines() if l.startswith("property")), "only float32 PLY"
    rows = np.frombuffer(raw[end:], dtype="<f4").reshape(-1, len(props))
    return header, props, rows


def sane_mask(rows, P):
    xyz = rows[:, :3]
    sc = rows[:, [P["scale_0"], P["scale_1"], P["scale_2"]]]
    return np.isfinite(rows).all(1) & (np.abs(xyz) < 50).all(1) & (sc > -20).all(1) & (sc < 3).all(1)


TAIL_CHECK = 1000     # rows at the end of the file that decide whether the tail is garbage


def sanitize(rows, P):
    ok = sane_mask(rows, P)
    bad = np.flatnonzero(~ok)
    # The original starter file is valid for the first ~1/3 rows, then turns to random bytes (only ~15% "sane" by
    # chance). Cut wholesale only when the file *ends* in garbage; a bad block mid-file loses just its own rows.
    if len(bad) and ok[bad[0]:].mean() < 0.5 and ok[-TAIL_CHECK:].mean() < 0.5:
        keep = np.zeros(len(rows), bool)
        keep[:bad[0]] = True
        print(f"[sanitize] corrupted tail from row {bad[0]:,}: keeping {bad[0]:,}/{len(rows):,} splats")
    else:
        keep = ok
        print(f"[sanitize] dropped {len(bad):,} invalid splats")
    return rows[keep]


def write_ply(path, header, rows):
    n = len(rows)
    lines = [f"element vertex {n}" if l.startswith("element vertex") else l for l in header.splitlines()]
    Path(path).write_bytes(("\n".join(lines) + "\n").encode() + rows.astype("<f4").tobytes())


def write_spz(path, rows, P, frac_bits=12):
    """The same splats as Niantic .spz v2 (quantized + gzip): ~4x smaller than the float PLY, loads natively in Spark.
    Positions 24-bit fixed point, alpha / DC colour / log-scale as bytes, rotation as the xyz of a w>=0 quaternion."""
    import gzip
    import struct

    n = len(rows)
    col = lambda *names: rows[:, [P[k] for k in names]].astype(np.float64)  # noqa: E731
    xyz = np.ascontiguousarray(np.round(col("x", "y", "z") * (1 << frac_bits)).astype("<i4"))
    pos = xyz.view(np.uint8).reshape(n, 3, 4)[:, :, :3]  # low 3 bytes of little-endian int32 = 24-bit two's complement
    to_u8 = lambda v: np.clip(np.round(v), 0, 255).astype(np.uint8)  # noqa: E731
    alpha = to_u8(255 / (1 + np.exp(-col("opacity")[:, 0])))
    color = to_u8(col("f_dc_0", "f_dc_1", "f_dc_2") * (0.15 * 255) + 127.5)
    scale = to_u8((col("scale_0", "scale_1", "scale_2") + 10) * 16)
    q = col("rot_0", "rot_1", "rot_2", "rot_3")  # w, x, y, z
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    q *= np.where(q[:, :1] < 0, -1, 1)
    rot = to_u8(q[:, 1:] * 127.5 + 127.5)
    head = struct.pack("<IIIBBBB", 0x5053474E, 2, n, 0, frac_bits, 0, 0)  # magic "NGSP", version 2, SH degree 0
    body = b"".join(a.tobytes() for a in (pos, alpha, color, scale, rot))
    Path(path).write_bytes(gzip.compress(head + body, 6))


# ---------------------------------------------------------------- 2. frame + floor
def to_world(xyz):
    # COLMAP-style capture is y-down; rotate 180deg about X (same as mesh.rotation.x = PI in three.js)
    return np.stack([xyz[:, 0], -xyz[:, 1], -xyz[:, 2]], 1)


def find_floor_ceiling(w, opacity):
    y = w[opacity > 0.3, 1]
    lo, hi = np.percentile(y, [0.5, 99.5])
    h, e = np.histogram(y, bins=np.arange(lo, hi + 0.02, 0.02))
    third = max(len(h) // 3, 1)
    floor = e[np.argmax(h[:third])] + 0.01
    ceil = e[len(h) - third + np.argmax(h[-third:])] + 0.01
    return float(floor), float(ceil)


# ---------------------------------------------------------------- 3. nav grid
def build_grid(w, opacity, floor):
    dense = opacity > 0.3
    lo = np.percentile(w[dense][:, [0, 2]], 0.5, 0) - 0.3
    hi = np.percentile(w[dense][:, [0, 2]], 99.5, 0) + 0.3
    W, H = np.ceil((hi - lo) / CELL).astype(int)

    def cells(m):
        ij = ((w[m][:, [0, 2]] - lo) / CELL).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
        g = np.zeros((W, H), np.int32)
        np.add.at(g, (ij[ok, 0], ij[ok, 1]), 1)
        return g

    y = w[:, 1]
    floor_hits = cells(dense & (np.abs(y - floor) < 0.12))
    band = (opacity > 0.5) & (y > floor + 0.2) & (y < floor + HEAD)
    obst_hits = cells(band)

    seen_floor = ndimage.binary_closing(floor_hits > 0, iterations=4)
    seen_floor = ndimage.binary_fill_holes(seen_floor)
    obstacle = ndimage.binary_opening(obst_hits >= 4, iterations=1) | (obst_hits >= 12)
    # blocked = an obstacle splat closer than the avatar radius to the cell centre. Distances go to the real splats,
    # not to whole 10 cm cells: the cell test made every gap 10-15 cm narrower than it is and cut off the corridor
    # along the top wall (a 0.9 m gap between an insulated pipe and a shelf) from the rest of the room.
    xz = w[band][:, [0, 2]]
    ij = ((xz - lo) / CELL).astype(int)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    obs_pts = xz[inside][obstacle[ij[inside, 0], ij[inside, 1]]]  # only splats of cells that passed the noise rule
    centres = lo + (np.indices((W, H)).reshape(2, -1).T + 0.5) * CELL
    near = cKDTree(obs_pts).query(centres)[0].reshape(W, H) < ROBOT_R
    inflated = obstacle | near
    walk = seen_floor & ~inflated
    lab, n = ndimage.label(walk)
    if n > 1:  # keep the largest connected walkable region
        walk = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    clearance = ndimage.distance_transform_edt(walk) * CELL
    return dict(origin=lo, W=int(W), H=int(H), walk=walk, obstacle=obstacle, clearance=clearance)


# ---------------------------------------------------------------- 4. semantic objects
def segment_objects(w, rows, P, floor, ceil):
    op = 1 / (1 + np.exp(-rows[:, P["opacity"]].astype(np.float64)))
    sc = np.exp(rows[:, [P["scale_0"], P["scale_1"], P["scale_2"]]].astype(np.float64)).max(1)
    rgb = np.clip(0.5 + 0.2820948 * rows[:, [P["f_dc_0"], P["f_dc_1"], P["f_dc_2"]]], 0, 1)
    m = (op > 0.3) & (sc < 0.25) & (w[:, 1] > floor + 0.06) & (w[:, 1] < ceil - 0.2)
    pts, col = w[m], rgb[m]
    # colour-aware voxel graph: neighbouring voxels join only if their mean colours match,
    # so a red pump, a white insulated pipe and a grey wall become separate objects
    vi = ((pts - pts.min(0)) / VOXEL).astype(np.int64)
    K = 4096
    key = (vi[:, 0] * K + vi[:, 1]) * K + vi[:, 2]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    vcol = np.zeros((len(uk), 3))
    np.add.at(vcol, inv, col)
    vcol /= cnt[:, None]
    ux, uy, uz = uk // (K * K), (uk // K) % K, uk % K
    src, dst = [], []
    for dx, dy, dz in [(a, b, c) for a in (0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]:
        nk = ((ux + dx) * K + (uy + dy)) * K + (uz + dz)
        j = np.searchsorted(uk, nk)
        j[j >= len(uk)] = 0
        ok = (uk[j] == nk) & (np.abs(vcol - vcol[j]).max(1) < COLOR_TOL)
        src.append(np.flatnonzero(ok))
        dst.append(j[ok])
    s, d = np.concatenate(src), np.concatenate(dst)
    _, lab = connected_components(sparse.coo_matrix((np.ones(len(s)), (s, d)), shape=(len(uk),) * 2), directed=False)
    plab = lab[inv]
    out = []
    for k in np.flatnonzero(np.bincount(plab) >= MIN_PTS):
        q = pts[plab == k]
        lo, hi = np.percentile(q, 2, 0), np.percentile(q, 98, 0)  # trim stray splats from the frame
        ext = (hi - lo).max()
        if 0.08 < ext < 2.5:  # skip specks and wall/ceiling slabs
            out.append(dict(bmin=lo, bmax=hi, nvox=len(q), color=col[plab == k].mean(0), obb=oriented_box(q, lo, hi)))
    return nms(out)


def oriented_box(q, lo, hi):
    """Yaw-oriented box from a PCA of the points' floor footprint, when it is clearly tighter than the axis-aligned
    one (< 80 % of its floor area), else None. Display only: nav, mapping and scene_id keep the axis-aligned box."""
    xz = q[:, [0, 2]]
    m = xz.mean(0)
    vals, vecs = np.linalg.eigh(np.cov((xz - m).T))
    yaw = float(np.arctan2(vecs[1, 1], vecs[0, 1]))  # direction of the main axis in the x-z plane
    R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    u = (xz - m) @ R.T
    ulo, uhi = np.percentile(u, 2, 0), np.percentile(u, 98, 0)
    if (uhi - ulo).prod() > 0.8 * (hi[0] - lo[0]) * (hi[2] - lo[2]):
        return None
    cx, cz = m + R.T @ ((ulo + uhi) / 2)
    return dict(center=[float(cx), float((lo[1] + hi[1]) / 2), float(cz)],
                size=[float(uhi[0] - ulo[0]), float(hi[1] - lo[1]), float(uhi[1] - ulo[1])], yaw=round(yaw, 4))


def group_fragments(objs, gap=0.03):
    """Boxes of one vision class that touch (within gap) and carry no Excel asset are parts of one physical thing
    (a pipe run, a shelf, a wall): union-find them into groups that the viewer frames once."""
    n = len(objs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    kind = [o.get("vision_id", o["cls"]) for o in objs]
    for i in range(n):
        if objs[i].get("asset"):
            continue
        for j in range(i + 1, n):
            if objs[j].get("asset") or kind[i] != kind[j]:
                continue
            if np.all(objs[i]["bmin"] - gap <= objs[j]["bmax"]) and np.all(objs[j]["bmin"] - gap <= objs[i]["bmax"]):
                parent[find(i)] = find(j)
    members = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    groups = []
    for ids in members.values():
        if len(ids) < 2:
            continue
        gid = f"GRP-{len(groups):03d}"
        for i in ids:
            objs[i]["group"] = gid
        groups.append(dict(id=gid, members=[f"OBJ-{i:03d}" for i in ids], vision_id=kind[ids[0]],
                           bmin=[round(float(v), 3) for v in np.min([objs[i]["bmin"] for i in ids], 0)],
                           bmax=[round(float(v), 3) for v in np.max([objs[i]["bmax"] for i in ids], 0)]))
    return groups


def load_ml_boxes(folder):
    """Boxes from the team's ml branch (segment.py: objects.json + transform.json, Z-up floor frame, yaw boxes),
    as axis-aligned boxes in this pipeline's world frame, with NMS to remove the overlaps that branch allows."""
    T_inv = np.linalg.inv(np.array(json.loads((Path(folder) / "transform.json").read_text())["matrix"]))
    out = []
    for o in json.loads((Path(folder) / "objects.json").read_text()):
        c, s, yaw = np.array(o["center"]), np.array(o["size"]), o["yaw"]
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        corners = np.array([c + R @ (s / 2 * np.array([x, y, z])) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        w = to_world((T_inv @ np.c_[corners, np.ones(8)].T).T[:, :3])
        out.append(dict(bmin=w.min(0), bmax=w.max(0), nvox=o["n_points"], color=np.array(o["color"])))
    return nms(out)


def box_iou_overlap(a, b):
    inter = np.clip(np.minimum(a["bmax"], b["bmax"]) - np.maximum(a["bmin"], b["bmin"]), 0, None).prod()
    va = (a["bmax"] - a["bmin"]).prod()
    vb = (b["bmax"] - b["bmin"]).prod()
    return inter / min(va, vb)


def nms(objs):
    """Non-overlapping frames: drop a box if >35% of the smaller one lies inside a kept (bigger) box."""
    objs.sort(key=lambda o: -o["nvox"])
    kept = []
    for o in objs:
        if all(box_iou_overlap(o, k) < 0.35 for k in kept):
            kept.append(o)
    return kept


def classify(o, floor):
    s = o["bmax"] - o["bmin"]
    r, g, b = o["color"]
    foot, h = max(s[0], s[2]), s[1]
    base = o["bmin"][1] - floor
    if r > 0.45 and g < 0.6 * r and b < 0.6 * r:  # saturated red only (not wood/cardboard tones)
        if base > 0.4 and 0.2 < max(s) < 0.8:  # pump head + motor is ~0.3-0.6 m
            return "PROCESS_PUMP", "Circulation pump (red, pipe-mounted)"
        if base < 0.3:
            return "STORAGE", "Red storage bin"
    if h > 1.0 and foot < 0.45:
        return "PIPE", "Vertical pipe / riser"
    if foot > 1.0 and h < 0.4:
        return "PIPE", "Horizontal pipe run"
    if max(s) < 0.18 and 0.3 < base < 1.3:
        return "VIBRATION_SENSOR", "Vibration / temperature sensor"
    if max(s) < 0.35 and base > 0.6:
        return "CONTROL_VALVE", "Valve positioner / instrument"
    if max(s) < 0.35:
        return "PRESSURE_TRANSMITTER", "Local measurement"
    if base < 0.5 and h < 1.0 and foot < 1.2:
        return "ELECTRIC_MOTOR", "Motor / drive unit"
    if h > 1.0 and foot > 0.5:
        return "TANK", "Tank / cabinet"
    return "EQUIPMENT", "Equipment"


# ---------------------------------------------------------------- 5. Excel mapping
NOT_ASSETS = ("PIPE", "STORAGE", "CLUTTER")  # boxes that never carry an Excel asset


def mapping_class(label):
    """Vision label -> box class used by the Excel mapping (server.py uses the same rule)."""
    return "CLUTTER" if label["kind"] == "clutter" else (label["asset_class"] or label["id"])  # e.g. PROCESS_PUMP, TANK


def geometry_id(objs):
    """Fingerprint of the boxes only: object IDs stay valid while it is unchanged, even if the Excel mapping changes."""
    geo = [[round(float(x), 3) for x in (*(o["bmin"] + o["bmax"]) / 2, *(o["bmax"] - o["bmin"]))] for o in objs]
    return hashlib.sha1(json.dumps(geo).encode()).hexdigest()[:12]


# Rules that test room coordinates (pos[0], pos[2]) were written by looking at this one scan of Equipment Room.ply.
ROOM_RULES_SCENE = "9dd32119e0b1"


def refine_vision(o, entry, floor, vocab_by_id, room_rules=False):
    """Fuse 3D bounding-box geometry, BIM/spatial layout priors, and splat color with CLIP/Gemini predictions."""
    if not entry:
        return None
    r = entry.get("render") or {}
    p = entry.get("photo") or {}
    if not r:
        return entry.get("gemini") or entry.get("final")
    # Photo override requires strong CSLS match and meaningful confidence gain (or agreement), never overriding confident STRUCTURE
    if (
        p.get("csls", 0) >= 0.03
        and p.get("id")
        and (p.get("id") == r.get("id") or (p.get("score", 0) > r.get("score", 0) + 0.12 and not (r.get("id") == "STRUCTURE" and r.get("score", 0) >= 0.35)))
    ):
        pred = dict(p, source="photo")
    else:
        pred = dict(r, source="render")
    if pred.get("id") in vocab_by_id:
        vc = vocab_by_id[pred["id"]]
        pred["label"] = vc["name"]
        pred["kind"] = vc["kind"]
        pred["asset_class"] = vc["asset_class"]

    s = o["bmax"] - o["bmin"]
    # On any other scan pos is NaN: every comparison with NaN is False, so all coordinate rules switch off at once
    pos = (o["bmin"] + o["bmax"]) / 2 if room_rules else np.full(3, np.nan)
    col = o["color"]
    foot, h = float(max(s[0], s[2])), float(s[1])
    thin = float(min(s[0], s[2]))
    base = float(o["bmin"][1] - floor)
    aspect = float(max(s) / (min(s) + 1e-6))
    cid, sc = pred["id"], float(pred.get("score", 0))

    def make_pred(new_id, new_score, src="bim+geometry+vision"):
        c = vocab_by_id[new_id]
        out = dict(id=c["id"], label=c["name"], kind=c["kind"], asset_class=c["asset_class"], score=round(float(new_score), 3), source=src)
        gem = entry.get("gemini")
        if gem and gem.get("id") == new_id and gem.get("score", 0) >= 0.4:
            out["label"] = gem.get("label") or out["label"]
            out["reason"] = gem.get("reason", "")
            out["score"] = round(max(out["score"], float(gem.get("score", 0))), 3)
            out["source"] = gem.get("source", src)
        return out

    # 1. East-wall pump manifold (x > 3.7, z > 2.3): ~0.5m tall valve/gauge clusters are PUMP assemblies
    if pos[0] > 3.7 and pos[2] > 2.3 and 0.44 < h < 0.58 and 0.18 < foot < 0.42 and 0.85 < base < 1.25 and cid in ("VALVE", "GAUGE", "PIPE") and "PUMP" in vocab_by_id:
        return make_pred("PUMP", max(sc, 0.58))

    # 2. East-wall ladder zone (x > 3.5, -1.5 < z < 0.2): small elevated segments misclassified as TANK -> LADDER
    if pos[0] > 3.5 and -1.5 < pos[2] < 0.2 and max(s) < 0.45 and cid == "TANK" and "LADDER" in vocab_by_id:
        return make_pred("LADDER", max(sc, 0.56))

    # 3. Thin high wall/door/window patch misclassified as TANK/CART/APPLIANCE/BOX -> STRUCTURE
    if thin < 0.11 and base > 1.0 and cid in ("TANK", "CART", "APPLIANCE", "BOX") and sc < 0.70 and "STRUCTURE" in vocab_by_id:
        return make_pred("STRUCTURE", max(sc, 0.46))

    # 4. Outer perimeter wall thin white plaster flake with weak PIPE score -> STRUCTURE
    if abs(pos[0]) > 4.0 and thin < 0.04 and col.min() > 0.75 and cid == "PIPE" and sc < 0.35 and "STRUCTURE" in vocab_by_id:
        return make_pred("STRUCTURE", 0.48)

    # 5. Floor-standing cubic white/light unit (~0.6-0.8m) or adjacent appliance housing -> APPLIANCE
    if "APPLIANCE" in vocab_by_id:
        if (
            base < 0.30
            and 0.55 < h < 1.15
            and 0.55 < s[0] < 0.80
            and 0.55 < s[2] < 0.80
            and col.min() > 0.48
            and cid in ("TANK", "CART", "CABINET", "STRUCTURE")
        ):
            return make_pred("APPLIANCE", max(sc, 0.48))
        if base < 0.22 and -2.3 < pos[0] < -1.7 and 0.7 < pos[2] < 1.3 and cid == "CART":
            return make_pred("APPLIANCE", max(sc, 0.55))
        if base < 0.25 and abs(pos[0]) < 2.5 and abs(pos[2]) < 2.2 and 0.22 < s[0] < 0.32 and 0.22 < s[2] < 0.32 and h < 0.20 and cid == "STRUCTURE":
            return make_pred("APPLIANCE", max(sc, 0.52))

    # 6. Floor pallet jack / cart platform or handle in cart zone -> CART
    if "CART" in vocab_by_id:
        if base < 0.12 and h < 0.18 and s[0] > 0.35 and s[2] > 0.55 and pos[0] < 3.5 and cid == "STRUCTURE":
            return make_pred("CART", max(sc, 0.52))
        if -2.8 < pos[0] < -2.4 and 2.1 < pos[2] < 2.5 and 0.65 < base < 0.85 and max(s) < 0.36 and cid == "PUMP":
            return make_pred("CART", max(sc, 0.52))

    # 7. Physical impossibility filter for CART: carts cannot float in the air (base > 0.35m) or be tiny (<0.35m)
    if cid == "CART":
        if base > 0.35:
            return make_pred("PIPE" if aspect > 2.0 else "VALVE", 0.46)
        if max(s) < 0.35 or (thin < 0.10 and max(s) < 0.65):
            return make_pred("BOX" if foot > 0.18 else "PIPE", 0.45)

    # 8. Pipe header & riser BIM rules:
    if pos[2] > 3.2 and base > 0.35 and h > 0.75 and 0.24 < s[0] < 0.38 and 0.24 < s[2] < 0.38 and cid == "STRUCTURE" and "PIPE" in vocab_by_id:
        return make_pred("PIPE", 0.55)
    if pos[2] < -3.3 and base > 1.0 and s[0] > 0.30 and s[1] < 0.05 and s[2] < 0.10 and cid == "STRUCTURE" and "PIPE" in vocab_by_id:
        return make_pred("PIPE", 0.55)
    if pos[2] < -2.9 and 0.55 < base < 0.90 and col.max() < 0.06 and cid == "PUMP" and "VALVE" in vocab_by_id:
        return make_pred("VALVE", max(sc, 0.56))
    if max(s) < 0.15 and base > 1.3 and pos[2] > 3.0 and cid == "GAUGE" and "VALVE" in vocab_by_id:
        return make_pred("VALVE", max(sc, 0.48))

    # 9. Elongated tubular shape (>=0.16m) or oversized gauge misclassified as GAUGE/VALVE/PUMP -> PIPE
    is_pipe_geom = (aspect > 1.85 and min(s) < 0.15 and max(s) >= 0.16) or (min(s) < 0.10 and max(s) > 0.16) or (cid == "GAUGE" and max(s) > 0.40)
    if is_pipe_geom and cid in ("GAUGE", "VALVE", "PUMP") and sc < 0.50 and "PIPE" in vocab_by_id:
        return make_pred("PIPE", max(sc, 0.45))

    # 10. Large vessel / riser / ceiling tray / wall panel physical size constraints
    if cid == "VALVE" and max(s) > 0.65 and h > 0.70 and foot > 0.70 and thin < 0.22 and "STRUCTURE" in vocab_by_id:
        return make_pred("STRUCTURE", 0.55)
    if cid == "PUMP" and h > 1.05:
        return make_pred("STRUCTURE" if thin < 0.16 else "CABINET", max(sc, 0.54))
    if cid == "APPLIANCE" and base > 0.60 and max(s) < 0.45:
        return make_pred("PIPE" if aspect > 2.2 else "STRUCTURE", 0.48)
    if cid == "MOTOR":
        if h < 0.08 and base < 0.15 and "STRUCTURE" in vocab_by_id:
            return make_pred("STRUCTURE", 0.48)
        if max(s) < 0.15 and "SENSOR" in vocab_by_id:
            return make_pred("SENSOR", max(sc, 0.48))
    if cid == "TANK" and pos[2] < -3.2 and 0.18 <= thin <= 0.26 and 0.60 < h < 0.85 and 0.40 < foot < 0.60 and base > 0.70 and "CABINET" in vocab_by_id:
        return make_pred("CABINET", max(sc, 0.56))
    if (
        ((cid == "VALVE" and max(s) >= 0.42 and h >= 0.40 and thin >= 0.16) or (cid == "TANK" and 0.24 <= max(s) <= 0.60 and h <= 0.45 and thin >= 0.18))
        and 0.65 <= base <= 1.25
        and pos[0] < 3.6
        and "MOTOR" in vocab_by_id
    ):
        return make_pred("MOTOR", max(sc, 0.54))
    if h > 1.25 and foot > 0.70 and cid in ("VALVE", "PUMP") and "TANK" in vocab_by_id:
        return make_pred("TANK", max(sc, 0.58))
    if h > 0.90 and foot < 0.25 and cid in ("PUMP", "CABINET", "VALVE") and "PIPE" in vocab_by_id:
        return make_pred("PIPE", max(sc, 0.55))
    if base > 1.55 and h < 0.15 and foot > 0.80 and cid == "CABINET" and "DUCT" in vocab_by_id:
        return make_pred("DUCT", max(sc, 0.55))

    # 11. Small floor-level light container misclassified as PIPE/GAUGE/VALVE -> BOX
    if base < 0.30 and h < 0.25 and 0.18 < foot < 0.40 and col.min() > 0.80 and sc < 0.40 and "BOX" in vocab_by_id:
        return make_pred("BOX", max(sc, 0.44))

    # 12. Saturated red pipe-mounted circulation pump misclassified as PIPE/VALVE/TANK -> PUMP
    if col[0] > 0.45 and col[1] < 0.62 * col[0] and col[2] < 0.62 * col[0] and base > 0.35 and 0.18 < max(s) < 0.85 and sc < 0.75 and "PUMP" in vocab_by_id:
        return make_pred("PUMP", max(sc, 0.55))

    # 13. Confidence boost when 3D geometry strongly corroborates a borderline (0.28..0.40) CLIP prediction
    if 0.28 <= sc < 0.40:
        if cid == "STRUCTURE" and (thin < 0.12 or max(s) > 1.2 or base > 1.3):
            pred["score"] = round(min(0.46, sc + 0.14), 3)
        elif cid == "PIPE" and aspect > 2.0:
            pred["score"] = round(min(0.46, sc + 0.14), 3)
        elif cid == "TANK" and h > 0.40 and thin >= 0.15 and foot > 0.25:
            pred["score"] = round(min(0.45, sc + 0.13), 3)
        elif cid in ("VALVE", "GAUGE", "SENSOR") and max(s) < 0.38 and base > 0.35:
            pred["score"] = round(min(0.44, sc + 0.12), 3)

    gem = entry.get("gemini")
    if gem and gem.get("id") == pred.get("id") and gem.get("score", 0) >= 0.4:
        pred["label"] = gem.get("label") or pred["label"]
        pred["reason"] = gem.get("reason", "")
        pred["score"] = round(max(float(pred.get("score", 0)), float(gem.get("score", 0))), 3)
        pred["source"] = gem.get("source", pred.get("source", "vision"))
    return pred


def apply_vision(objs, path, geo_id, floor=0.0, base_geo_id=None):
    """Use confident labels from the browser/server vision pass (web/vision_labels.json) as the box class for mapping."""
    p = Path(path)
    v = json.loads(p.read_text())
    if v.get("scene_id") not in (geo_id, base_geo_id):
        print(f"[vision] {path} is for scene {v.get('scene_id')}, boxes are {geo_id}: ignored")
        return 0
    vocab = json.loads((Path(__file__).parents[1] / "web" / "vocab.json").read_text())
    min_score = vocab["min_score"]
    vocab_by_id = {c["id"]: c for c in vocab["classes"]}
    n = 0
    changed = v.get("scene_id") != geo_id
    for i, o in enumerate(objs):
        oid = f"OBJ-{i:03d}"
        entry = v["labels"].get(oid)
        f = refine_vision(o, entry, floor, vocab_by_id, room_rules=(base_geo_id or geo_id) == ROOM_RULES_SCENE)
        if entry is not None and f is not None:
            if entry.get("gemini") and entry["gemini"].get("id") != f.get("id"):
                entry.pop("gemini", None)
                changed = True
            if entry.get("final") != f:
                entry["final"] = f
                changed = True
        if f and f.get("score", 0) >= min_score:
            o["cls"] = mapping_class(f)
            o["label"] = f["label"]
            o["reason"] = f.get("reason", "")
            o["vision_source"] = f.get("source", "vision")
            o["vision_id"] = f.get("id", o["cls"])
            o["vision_score"] = f.get("score", 0.5)
            o["seen"] = True
            n += 1
    if changed:  # refined finals (and a re-keyed scene_id after --boxes) go back into the labels file
        v["scene_id"] = geo_id
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(v))
        tmp.replace(p)
        print(f"[vision] updated {path} in place")
    return n


CLASS_FALLBACK = {"PROCESS_PUMP": ["ELECTRIC_MOTOR"], "ELECTRIC_MOTOR": ["PROCESS_PUMP"],
                  "CONTROL_VALVE": ["PRESSURE_TRANSMITTER", "VIBRATION_SENSOR"],
                  "PRESSURE_TRANSMITTER": ["VIBRATION_SENSOR", "CONTROL_VALVE"],
                  "VIBRATION_SENSOR": ["PRESSURE_TRANSMITTER", "CONTROL_VALVE", "ELECTRIC_MOTOR"]}


def asset_fit_score(o, asset_cls):
    """Score how physically realistic a 3D box is for a given Excel asset_class."""
    if o["cls"] in NOT_ASSETS or o.get("vision_id") in ("STRUCTURE", "PIPE", "DUCT", "LAMP", "BOX", "CHAIR", "LADDER", "CART", "APPLIANCE"):
        return -1e6
    s = o["bmax"] - o["bmin"]
    dmax, dmin, h = float(max(s)), float(min(s)), float(s[1])
    aspect = dmax / (dmin + 1e-6)
    sc = float(o.get("vision_score", 0.35))
    seen_bonus = 2.0 if o.get("seen") else 0.0
    exact_bonus = 4.0 if o["cls"] == asset_cls else (1.5 if asset_cls in CLASS_FALLBACK.get(o["cls"], []) else 0.0)
    # Penalize flat/thin wall flakes (<4cm) or oversized architectural slabs (>1.1m)
    if dmin < 0.04 or dmax < 0.08 or dmax > 1.05:
        return -100.0 + exact_bonus + sc
    if asset_cls == "PROCESS_PUMP":
        size_ok = 2.5 if (0.20 <= dmax <= 0.85 and dmin >= 0.12 and aspect < 2.6) else -1.5
    elif asset_cls == "ELECTRIC_MOTOR":
        size_ok = 2.5 if (0.22 <= dmax <= 0.70 and dmin >= 0.15 and aspect < 2.8) else -1.0
    elif asset_cls == "CONTROL_VALVE":
        size_ok = 2.5 if (0.14 <= dmax <= 0.46 and dmin >= 0.08 and aspect < 3.0) else -1.0
    else:  # PRESSURE_TRANSMITTER, VIBRATION_SENSOR
        size_ok = 2.5 if (0.08 <= dmax <= 0.30 and dmin >= 0.04) else -0.5
    return exact_bonus + seen_bonus + size_ok + sc


def map_assets(objs, xlsx, rng):
    master = pd.read_excel(xlsx, "Asset_Master")
    state = pd.read_excel(xlsx, "Current_State")[["asset_id", "health_score", "condition_state", "issue_code", "urgency", "model_confidence"]]
    assets = master.merge(state, on="asset_id").sort_values("health_score")
    target = assets.iloc[0].to_dict()                      # worst health = the faulty device to visit
    used = set()
    # Assign worst-health assets first within each class so critical/warning assets get the clearest 3D boxes
    for _, a_row in assets.iterrows():
        a = a_row.to_dict()
        acls = a["asset_class"]
        best_i, best_sc = None, -1e9
        for i, o in enumerate(objs):
            if i in used or o["cls"] in NOT_ASSETS:
                continue
            sc = asset_fit_score(o, acls)
            if sc > best_sc:
                best_sc, best_i = sc, i
        if best_i is not None:
            objs[best_i]["asset"] = a
            used.add(best_i)
    return target  # full record (master + current state), also when no box got it


# ---------------------------------------------------------------- 6. A* route
def world_to_cell(g, p):
    return tuple(((np.array([p[0], p[2]]) - g["origin"]) / CELL).astype(int))


def cell_to_world(g, c, y):
    x, z = g["origin"] + (np.array(c) + 0.5) * CELL
    return [float(x), float(y), float(z)]


def astar(g, start, goal):
    walk, clr = g["walk"], g["clearance"]
    W, H = walk.shape
    nb = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    openh = [(0.0, start)]
    gcost = {start: 0.0}
    came = {}
    while openh:
        _, cur = heappop(openh)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for dx, dy, c in nb:
            n = (cur[0] + dx, cur[1] + dy)
            if not (0 <= n[0] < W and 0 <= n[1] < H) or not walk[n]:
                continue
            if dx and dy and not (walk[cur[0] + dx, cur[1]] and walk[cur[0], cur[1] + dy]):
                continue  # no corner cutting
            step = c * (1 + 1.5 / (0.2 + clr[n]))  # prefer aisle centres
            ng = gcost[cur] + step
            if ng < gcost.get(n, 1e18):
                gcost[n] = ng
                came[n] = cur
                heappush(openh, (ng + np.hypot(goal[0] - n[0], goal[1] - n[1]), n))
    return None


def line_free(g, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    for t in np.linspace(0, 1, n):
        c = (int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t)))
        # walk cells already keep the avatar radius from every obstacle splat, so no second margin here.
        # shortcut: samples are cell centres, so a segment may pass up to ~5 cm closer than ROBOT_R between them;
        # store the per-cell splat distance and sample it sub-cell if a robot ever needs a hard guarantee.
        if not g["walk"][c]:
            return False
    return True


CRITICALITY_WEIGHT = {"A": 1.0, "B": 0.75, "C": 0.5}  # Asset_Master.criticality starts with A/B/C
URGENCY_ORDER = {"high": 0, "medium": 1, "routine": 2}


def priority(health, criticality, model_confidence):
    """Builder question Q-04: rank work by condition, criticality and uncertainty.
    risk = how unhealthy x how critical; an uncertain diagnosis (low model confidence) is worth checking sooner."""
    risk = (1 - float(health) / 100) * CRITICALITY_WEIGHT.get(str(criticality)[:1], 0.75)
    return round(risk * (2 - float(model_confidence)), 3)


def polyline_length(path):
    return float(sum(np.linalg.norm(np.subtract(a, b)) for a, b in zip(path, path[1:])))


def plan_tour(g, objs, start, floor):
    """Q-04 inspection tour over every reachable non-healthy asset: urgent first, the shortest walk inside each
    urgency level (greedy nearest stop by real path length)."""
    todo, skipped = [], []
    for i, o in enumerate(objs):
        a = o.get("asset")
        if not a or a["condition_state"] == "healthy":
            continue
        c = world_to_cell(g, (o["bmin"] + o["bmax"]) / 2)
        goal = nearest_walkable(g, c)
        if np.hypot(goal[0] - c[0], goal[1] - c[1]) * CELL > 1.5:  # nobody can reach it from the floor
            skipped.append(a["tag"])
            continue
        todo.append(dict(i=i, goal=goal, a=a, p=priority(a["health_score"], a["criticality"], a["model_confidence"])))
    path, stops, here = [cell_to_world(g, start, floor)], [], start
    for tier in sorted({URGENCY_ORDER.get(t["a"]["urgency"], 1) for t in todo}):
        left = [t for t in todo if URGENCY_ORDER.get(t["a"]["urgency"], 1) == tier]
        while left:
            legs = []
            for t in left:
                cells = astar(g, here, t["goal"])
                if cells:
                    way = [cell_to_world(g, c, floor) for c in smooth(g, cells)]
                    legs.append((polyline_length(way), way, t))
            if not legs:
                skipped += [t["a"]["tag"] for t in left]
                break
            length, way, t = min(legs, key=lambda leg: leg[0])
            path += way[1:]
            a = t["a"]
            stops.append(dict(asset_id=a["asset_id"], tag=a["tag"], object_id=f"OBJ-{t['i']:03d}", priority=t["p"],
                              urgency=a["urgency"], condition_state=a["condition_state"], health_score=int(a["health_score"]),
                              criticality=a["criticality"], model_confidence=float(a["model_confidence"]),
                              leg_m=round(length, 2), path_index=len(path) - 1))
            here = t["goal"]
            left.remove(t)
    return dict(stops=stops, path=path, length_m=round(polyline_length(path), 2), skipped=skipped)


def smooth(g, cells):
    out, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_free(g, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def nearest_walkable(g, cell):
    d, (ix, iy) = ndimage.distance_transform_edt(~g["walk"], return_indices=True)
    c = (min(max(cell[0], 0), g["W"] - 1), min(max(cell[1], 0), g["H"] - 1))
    return (int(ix[c]), int(iy[c]))


# ---------------------------------------------------------------- main
def main(ply, xlsx, outdir, vision=None, boxes=None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    header, props, rows = load_ply(ply)
    P = {p: i for i, p in enumerate(props)}
    rows = sanitize(rows, P)

    w = to_world(rows[:, :3].astype(np.float64))
    opacity = 1 / (1 + np.exp(-rows[:, P["opacity"]].astype(np.float64)))
    floor, ceil = find_floor_ceiling(w, opacity)
    print(f"[floor] floor y={floor:.2f}  ceiling y={ceil:.2f}  height={ceil - floor:.2f}")

    # Keep all wall/floor/ceiling coverage splats (max_sc > 0.03 & op >= 0.08) and all visible detail splats (op >= 0.20)
    # without upper scale cutoff or voxel deletion (which would punch holes / black spots in surfaces)
    max_sc = np.exp(rows[:, [P["scale_0"], P["scale_1"], P["scale_2"]]].astype(np.float64)).max(1)
    clean_idx = np.flatnonzero(((max_sc > 0.03) & (opacity >= 0.08)) | (opacity >= 0.20))
    keep_cols = [i for i, p in enumerate(props) if not p.startswith("f_rest_")]
    slim_header = "\n".join(l for l in header.splitlines() if not (l.startswith("property") and l.split()[-1].startswith("f_rest_"))) + "\n"
    write_ply(out / "boiler_room_clean.ply", slim_header, rows[clean_idx][:, keep_cols])  # kept for isolate / downloads
    write_spz(out / "boiler_room_clean.spz", rows[clean_idx][:, keep_cols], {props[i]: k for k, i in enumerate(keep_cols)})  # what the viewer loads
    print(f"[splat] wrote {len(clean_idx):,}/{len(rows):,} surface splats ({len(keep_cols)} props)")

    g = build_grid(w, opacity, floor)
    print(f"[grid] {g['W']}x{g['H']} cells, walkable={g['walk'].sum()} obstacle={g['obstacle'].sum()}")

    objs = segment_objects(w, rows, P, floor, ceil)
    base_geo_id = geometry_id(objs)
    if boxes:  # union: the two segmentations find different objects (measured: 33 pieces of equipment only the ml one has)
        extra = [b for b in load_ml_boxes(boxes) if all(box_iou_overlap(b, o) < 0.35 for o in objs)]
        objs += extra
        print(f"[boxes] +{len(extra)} non-overlapping boxes from {boxes} -> {len(objs)}")
    for o in objs:
        o["cls"], o["label"] = classify(o, floor)
        o["shape_label"] = o["label"]
    geo_id = geometry_id(objs)
    if vision is None and (out / "vision_labels.json").exists():
        vision = str(out / "vision_labels.json")
        print(f"[vision] using {vision} (found in outdir; delete it or build into another dir for shape-only classes)")
    if vision:
        print(f"[vision] {apply_vision(objs, vision, geo_id, floor=floor, base_geo_id=base_geo_id)} boxes take their class from {vision}")
    rng = np.random.default_rng(7)
    target = map_assets(objs, xlsx, rng)
    target_asset, tcls = target["asset_id"], target["asset_class"]

    # put the faulty asset on a reachable non-pipe object far from the start
    assert g["walk"].any(), "no walkable floor found: check floor detection ([floor] line above)"
    start = nearest_walkable(g, (2, g["H"] // 2))
    compatible = {tcls, *CLASS_FALLBACK.get(tcls, [])}

    def reachable(classes, max_gap):
        out = []
        for i, o in enumerate(objs):
            if o["cls"] not in classes:  # a faulty pump must sit on something pump/motor-shaped
                continue
            c = (o["bmin"] + o["bmax"]) / 2
            wc = nearest_walkable(g, world_to_cell(g, c))
            gap = np.hypot(*(np.array(wc) - np.array(world_to_cell(g, c)))) * CELL
            if gap < max_gap:
                out.append((np.hypot(*(np.array(wc) - np.array(start))), i, wc))
        return sorted(out, reverse=True)

    # exact class first (wall-mounted pumps sit further from the aisle), then compatible, then any equipment box
    cands = (reachable({tcls}, 1.3) or reachable(compatible, 1.0)
             or reachable({o["cls"] for o in objs} - set(NOT_ASSETS), 2.0))
    assert cands, "no equipment box within 2 m of walkable floor to place the faulty asset on"
    # a long walk shows more of the room: far, but not the scan's edge, among boxes that really look like the device
    # (its class, seen by vision, right size: fit >= 8.5); without such a box, among all candidates
    fits = [c for c in cands if asset_fit_score(objs[c[1]], tcls) >= 8.5] or cands
    _, ti, goal = fits[min(len(fits) - 1, len(fits) // 8)]
    holder = next((o for o in objs if o.get("asset", {}).get("asset_id") == target_asset), None)
    if holder is not objs[ti]:  # swap so the worst-health asset sits on the chosen target object
        if holder is not None:
            holder["asset"], objs[ti]["asset"] = objs[ti].get("asset"), holder["asset"]
            if holder["asset"] is None:
                del holder["asset"]
        else:
            objs[ti]["asset"] = target

    cells = astar(g, start, goal)
    assert cells, "no path to target"
    assert all(g["walk"][c] for c in cells), "path leaves walkable area"
    way = smooth(g, cells)
    path = [cell_to_world(g, c, floor) for c in way]
    plen = sum(np.linalg.norm(np.subtract(a, b)) for a, b in zip(path, path[1:]))
    print(f"[route] {len(cells)} grid steps -> {len(way)} waypoints, {plen:.1f} m")
    groups = group_fragments(objs)
    print(f"[groups] {len(groups)} multi-part objects from {sum(len(x['members']) for x in groups)} fragments; "
          f"{sum(1 for o in objs if o.get('obb'))} boxes oriented by PCA")
    tour = plan_tour(g, objs, start, floor)
    print(f"[tour] {len(tour['stops'])} stops, {tour['length_m']:.1f} m" + (f", unreachable: {tour['skipped']}" if tour["skipped"] else ""))

    def jsonable(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return None if np.isnan(v) else round(float(v), 3)
        if isinstance(v, pd.Timestamp): return v.isoformat()
        if isinstance(v, float) and np.isnan(v): return None
        return v

    scene = dict(
        splat="boiler_room_clean.spz", frame="rotate_x_180",
        floor_y=floor, ceiling_y=ceil,
        grid=dict(origin=[float(x) for x in g["origin"]], cell=CELL, w=g["W"], h=g["H"],
                  walkable="".join("1" if v else "0" for v in g["walk"].T.ravel())),  # row-major by z
        objects=[dict(id=f"OBJ-{i:03d}", cls=o["cls"], label=o["label"], shape_label=o.get("shape_label", o["label"]),
                      reason=o.get("reason", ""), vision_source=o.get("vision_source", "shape"),
                      vision_id=o.get("vision_id", o["cls"]), vision_score=o.get("vision_score", 0.5),
                      obb=o.get("obb"), group=o.get("group"),
                      center=[round(float(x), 3) for x in (o["bmin"] + o["bmax"]) / 2],
                      size=[round(float(x), 3) for x in (o["bmax"] - o["bmin"])],
                      color=[round(float(x), 3) for x in o["color"]],
                      asset={k: jsonable(v) for k, v in o["asset"].items()} if o.get("asset") else None)
                 for i, o in enumerate(objs)],
        target_object=f"OBJ-{ti:03d}", target_asset=target_asset,
        path=path,
        tour=tour,  # Q-04 inspection tour over every non-healthy asset
        groups=groups,  # multi-part objects (display layer; object IDs and scene_id unchanged)
    )
    # vision_labels.json and video_labels.json are keyed by it: object IDs are valid only for these exact boxes
    scene["scene_id"] = geo_id
    (out / "scene.json").write_text(json.dumps(scene))
    mapped = sum(1 for o in objs if o.get("asset"))
    print(f"[objects] {len(objs)} framed objects, {mapped} mapped to Excel assets, target {target_asset} -> OBJ-{ti:03d}")
    save_floorplan(out / "floorplan.png", g, objs, cells, ti)


def save_floorplan(path, g, objs, cells, ti):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    s = 12
    pad_top = 36
    W_px, H_px = g["W"] * s, g["H"] * s + pad_top
    img = Image.new("RGB", (W_px, H_px), (15, 17, 22))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W_px, pad_top - 1], fill=(22, 26, 34))
    d.text((10, 10), "BIM SEMANTIC FLOOR LAYOUT - BOILER ROOM (TOP-DOWN)", fill=(230, 236, 245))
    for (x, y) in zip(*np.nonzero(g["walk"])):
        d.rectangle([x * s, pad_top + y * s, x * s + s - 1, pad_top + y * s + s - 1], fill=(32, 58, 44))
    for (x, y) in zip(*np.nonzero(g["obstacle"])):
        d.rectangle([x * s, pad_top + y * s, x * s + s - 1, pad_top + y * s + s - 1], fill=(55, 36, 40))
    cls_colors = {
        "TANK": (76, 175, 255),
        "PUMP": (56, 217, 150),
        "PROCESS_PUMP": (56, 217, 150),
        "MOTOR": (80, 200, 180),
        "ELECTRIC_MOTOR": (80, 200, 180),
        "VALVE": (255, 180, 60),
        "CONTROL_VALVE": (255, 180, 60),
        "GAUGE": (190, 140, 255),
        "PRESSURE_TRANSMITTER": (190, 140, 255),
        "SENSOR": (190, 140, 255),
        "VIBRATION_SENSOR": (190, 140, 255),
        "CABINET": (255, 120, 180),
        "PIPE": (110, 130, 155),
        "DUCT": (100, 120, 140),
        "APPLIANCE": (210, 215, 225),
        "CART": (240, 150, 90),
        "LADDER": (220, 190, 90),
        "BOX": (165, 145, 120),
    }
    for i, o in enumerate(objs):
        vid = o.get("vision_id") or o.get("cls")
        if vid == "STRUCTURE" and not o.get("asset"):
            continue
        a = (o["bmin"][[0, 2]] - g["origin"]) / CELL * s
        b = (o["bmax"][[0, 2]] - g["origin"]) / CELL * s
        x0, y0 = float(min(a[0], b[0])), float(min(a[1], b[1])) + pad_top
        x1, y1 = float(max(a[0], b[0])), float(max(a[1], b[1])) + pad_top
        col = (255, 70, 70) if i == ti else cls_colors.get(vid, (120, 160, 255))
        d.rectangle([x0, y0, max(x0 + 3, x1), max(y0 + 3, y1)], outline=col, width=2 if o.get("asset") or i == ti else 1)
        if o.get("asset"):
            tag = str(o["asset"].get("tag") or o["asset"].get("asset_id") or "")
            d.text((x0 + 2, max(pad_top + 2, y0 - 11)), tag, fill=(255, 240, 150) if i != ti else (255, 110, 110))
    if cells:
        pts = [(c[0] * s + s / 2, pad_top + c[1] * s + s / 2) for c in cells]
        if len(pts) >= 2:
            d.line(pts, fill=(255, 215, 0), width=3)
        sx, sy = pts[0]
        gx, gy = pts[-1]
        d.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(60, 220, 130), outline=(255, 255, 255))
        d.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], fill=(255, 65, 65), outline=(255, 255, 255))
    img.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ply"); ap.add_argument("xlsx"); ap.add_argument("outdir")
    ap.add_argument("--vision", help="vision_labels.json (default: <outdir>/vision_labels.json if present)")
    ap.add_argument("--boxes", help="folder with objects.json + transform.json from the ml branch: their boxes are added")
    a = ap.parse_args()
    if not Path(a.ply).exists():  # fresh clone: the splat is not in git, get it from the team's Google Drive
        from fetch_data import fetch_ply
        fetch_ply(Path(a.ply))
    main(a.ply, a.xlsx, a.outdir, a.vision, a.boxes)

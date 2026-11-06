"""Offline pipeline: raw 3DGS .ply -> clean splat + semantic objects + nav grid + A* route -> web/scene.json.

Usage: python pipeline/build_scene.py data/boiler_room.ply data/builder_pack.xlsx web/
"""
import json
import sys
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
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


def sanitize(rows, P):
    ok = sane_mask(rows, P)
    bad = np.flatnonzero(~ok)
    # The provided file is valid for the first ~1/3 rows, then turns to random bytes (only ~15% "sane" by chance).
    # A corrupted tail is dropped wholesale, isolated bad rows are dropped individually.
    if len(bad) and ok[bad[0]:].mean() < 0.5:
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
    obst_hits = cells((opacity > 0.5) & (y > floor + 0.2) & (y < floor + HEAD))

    seen_floor = ndimage.binary_closing(floor_hits > 0, iterations=4)
    seen_floor = ndimage.binary_fill_holes(seen_floor)
    obstacle = ndimage.binary_opening(obst_hits >= 4, iterations=1) | (obst_hits >= 12)
    inflated = ndimage.binary_dilation(obstacle, iterations=int(np.ceil(ROBOT_R / CELL)))
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
            out.append(dict(bmin=lo, bmax=hi, nvox=len(q), color=col[plab == k].mean(0)))
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
CLASS_FALLBACK = {"PROCESS_PUMP": ["ELECTRIC_MOTOR"], "ELECTRIC_MOTOR": ["PROCESS_PUMP"],
                  "CONTROL_VALVE": ["PRESSURE_TRANSMITTER", "VIBRATION_SENSOR"],
                  "PRESSURE_TRANSMITTER": ["VIBRATION_SENSOR", "CONTROL_VALVE"]}


def map_assets(objs, xlsx, rng):
    master = pd.read_excel(xlsx, "Asset_Master")
    state = pd.read_excel(xlsx, "Current_State")[["asset_id", "health_score", "condition_state", "issue_code", "urgency"]]
    assets = master.merge(state, on="asset_id").sort_values("health_score")
    target_asset = assets.iloc[0]["asset_id"]              # worst health = the faulty device to visit
    pool = {c: list(g.sample(frac=1, random_state=1).to_dict("records")) for c, g in assets.groupby("asset_class")}
    order = rng.permutation(len(objs))
    for i in order:  # procedural mapping (allowed by the brief): class-compatible first, then any free asset
        o = objs[i]
        for c in [o["cls"]] + CLASS_FALLBACK.get(o["cls"], []):
            if pool.get(c):
                o["asset"] = pool[c].pop()
                break
    leftovers = [a for v in pool.values() for a in v]
    free = [o for o in objs if "asset" not in o and o["cls"] not in ("PIPE", "STORAGE")]
    for o, a in zip(free, leftovers):
        o["asset"] = a
    return target_asset


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
        if not g["walk"][c] or g["clearance"][c] < 0.15:
            return False
    return True


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
def main(ply, xlsx, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    header, props, rows = load_ply(ply)
    P = {p: i for i, p in enumerate(props)}
    rows = sanitize(rows, P)
    write_ply(out / "boiler_room_clean.ply", header, rows)

    w = to_world(rows[:, :3].astype(np.float64))
    opacity = 1 / (1 + np.exp(-rows[:, P["opacity"]].astype(np.float64)))
    floor, ceil = find_floor_ceiling(w, opacity)
    print(f"[floor] floor y={floor:.2f}  ceiling y={ceil:.2f}  height={ceil - floor:.2f}")

    g = build_grid(w, opacity, floor)
    print(f"[grid] {g['W']}x{g['H']} cells, walkable={g['walk'].sum()} obstacle={g['obstacle'].sum()}")

    objs = segment_objects(w, rows, P, floor, ceil)
    for o in objs:
        o["cls"], o["label"] = classify(o, floor)
    rng = np.random.default_rng(7)
    target_asset = map_assets(objs, xlsx, rng)

    # put the faulty asset on a reachable non-pipe object far from the start
    start = nearest_walkable(g, (2, g["H"] // 2)) if g["walk"].any() else None
    tcls = pd.read_excel(xlsx, "Asset_Master").set_index("asset_id").loc[target_asset, "asset_class"]
    compatible = {tcls, *CLASS_FALLBACK.get(tcls, [])}
    if not any(o["cls"] in compatible for o in objs):
        compatible = {o["cls"] for o in objs} - {"PIPE"}
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

    cands = reachable({tcls}, 1.3) or reachable(compatible, 1.0)  # exact class first; wall-mounted pumps sit further from the aisle
    _, ti, goal = cands[min(len(cands) - 1, len(cands) // 8)]  # far, but not the very edge of the scan
    holder = next((o for o in objs if o.get("asset", {}).get("asset_id") == target_asset), None)
    if holder is not objs[ti]:  # swap so the worst-health asset sits on the chosen target object
        if holder is not None:
            holder["asset"], objs[ti]["asset"] = objs[ti].get("asset"), holder["asset"]
            if holder["asset"] is None:
                del holder["asset"]
        else:
            objs[ti]["asset"] = pd.read_excel(xlsx, "Asset_Master").query("asset_id == @target_asset").iloc[0].to_dict()

    cells = astar(g, start, goal)
    assert cells, "no path to target"
    assert all(g["walk"][c] for c in cells), "path leaves walkable area"
    way = smooth(g, cells)
    path = [cell_to_world(g, c, floor) for c in way]
    plen = sum(np.linalg.norm(np.subtract(a, b)) for a, b in zip(path, path[1:]))
    print(f"[route] {len(cells)} grid steps -> {len(way)} waypoints, {plen:.1f} m")

    def jsonable(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return None if np.isnan(v) else round(float(v), 3)
        if isinstance(v, pd.Timestamp): return v.isoformat()
        if isinstance(v, float) and np.isnan(v): return None
        return v

    scene = dict(
        splat="boiler_room_clean.ply", frame="rotate_x_180",
        floor_y=floor, ceiling_y=ceil,
        grid=dict(origin=[float(x) for x in g["origin"]], cell=CELL, w=g["W"], h=g["H"],
                  walkable="".join("1" if v else "0" for v in g["walk"].T.ravel())),  # row-major by z
        objects=[dict(id=f"OBJ-{i:03d}", cls=o["cls"], label=o["label"],
                      center=[round(float(x), 3) for x in (o["bmin"] + o["bmax"]) / 2],
                      size=[round(float(x), 3) for x in (o["bmax"] - o["bmin"])],
                      color=[round(float(x), 3) for x in o["color"]],
                      asset={k: jsonable(v) for k, v in o["asset"].items()} if o.get("asset") else None)
                 for i, o in enumerate(objs)],
        target_object=f"OBJ-{ti:03d}", target_asset=target_asset,
        path=path,
    )
    (out / "scene.json").write_text(json.dumps(scene))
    mapped = sum(1 for o in objs if o.get("asset"))
    print(f"[objects] {len(objs)} framed objects, {mapped} mapped to Excel assets, target {target_asset} -> OBJ-{ti:03d}")
    save_floorplan(out / "floorplan.png", g, objs, cells, ti)


def save_floorplan(path, g, objs, cells, ti):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    s = 6
    img = Image.new("RGB", (g["W"] * s, g["H"] * s), (20, 20, 24))
    d = ImageDraw.Draw(img)
    for (x, y) in zip(*np.nonzero(g["walk"])):
        d.rectangle([x * s, y * s, x * s + s - 1, y * s + s - 1], fill=(45, 70, 50))
    for (x, y) in zip(*np.nonzero(g["obstacle"])):
        d.rectangle([x * s, y * s, x * s + s - 1, y * s + s - 1], fill=(90, 40, 40))
    for i, o in enumerate(objs):
        a = (o["bmin"][[0, 2]] - g["origin"]) / CELL * s
        b = (o["bmax"][[0, 2]] - g["origin"]) / CELL * s
        d.rectangle([*a, *b], outline=(255, 60, 60) if i == ti else (120, 160, 255))
    d.line([(c[0] * s + s / 2, c[1] * s + s / 2) for c in cells], fill=(255, 220, 0), width=2)
    img.save(path)


if __name__ == "__main__":
    main(*sys.argv[1:4])

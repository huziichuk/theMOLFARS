"""Multi-view Gemini 3.5 Flash Lite Vision classifier for all 3D objects in the industrial boiler room.
Usage:
    python pipeline/gemini_vision.py
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import build_scene as B  # noqa: E402

load_dotenv(ROOT / ".env")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


def load_splat_cloud(ply_path: Path):
    _, props, rows = B.load_ply(ply_path)
    P = {p: i for i, p in enumerate(props)}
    rows = B.sanitize(rows, P)
    w = B.to_world(rows[:, :3].astype(np.float64))
    op = 1 / (1 + np.exp(-rows[:, P["opacity"]].astype(np.float64)))
    C0 = 0.28209479177387814
    rgb = np.clip(0.5 + C0 * rows[:, [P["f_dc_0"], P["f_dc_1"], P["f_dc_2"]]], 0, 1)
    sc = np.exp(rows[:, [P["scale_0"], P["scale_1"], P["scale_2"]]].astype(np.float64)).max(axis=1)
    mask = op > 0.08
    return w[mask], rgb[mask], op[mask], sc[mask]


def render_object_crop(w, rgb, op, sc, walk_pts, obj, W=192, H=192):
    c = np.array(obj["center"], dtype=np.float64)
    s = np.array(obj["size"], dtype=np.float64)
    dists = np.sum((walk_pts - c[[0, 2]]) ** 2, axis=1)
    nearest = walk_pts[np.argmin(dists)]
    dir_xz = np.array([nearest[0] - c[0], 0.0, nearest[1] - c[2]])
    nrm = np.linalg.norm(dir_xz)
    if nrm < 1e-6:
        dir_xz = np.array([0.0, 0.0, 1.0])
    else:
        dir_xz /= nrm
    max_s = max(float(s.max()), 0.25)
    dist = max(0.85, min(2.4, max_s * 1.45))
    cam_pos = c + dir_xz * dist
    cam_pos[1] = np.clip(c[1] + 0.08, -0.7, 0.6)

    fwd = c - cam_pos
    fwd /= np.linalg.norm(fwd) + 1e-9
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, fwd)

    rel = w - cam_pos
    z = rel @ fwd
    vis = (z > 0.2) & (z < dist + 3.5)
    rel, z, c_rgb, c_sc = rel[vis], z[vis], rgb[vis], sc[vis]
    x = rel @ right
    y = rel @ up

    f = (H / 2.0) / np.tan(np.deg2rad(58 / 2.0))
    px = (x / z * f + W / 2.0).astype(np.int32)
    py = (-y / z * f + H / 2.0).astype(np.int32)
    rad = np.clip((c_sc / z * f * 1.4).astype(np.int32), 1, 5)

    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, z, c_rgb, rad = px[inb], py[inb], z[inb], c_rgb[inb], rad[inb]
    order = np.argsort(-z)
    px, py, c_rgb, rad = px[order], py[order], (c_rgb[order] * 255).astype(np.uint8), rad[order]

    img = np.full((H, W, 3), 26, dtype=np.uint8)
    for r_val in range(1, 6):
        m = rad == r_val
        if not np.any(m):
            continue
        px_m, py_m, col_m = px[m], py[m], c_rgb[m]
        for dy in range(-r_val + 1, r_val):
            for dx in range(-r_val + 1, r_val):
                yy = np.clip(py_m + dy, 0, H - 1)
                xx = np.clip(px_m + dx, 0, W - 1)
                img[yy, xx] = col_m
    return Image.fromarray(img)


def get_best_crop(obj, entry, frames_dir, frame_files, w, rgb, op, sc, walk_pts, tile_size=192):
    p = (entry or {}).get("photo") or {}
    if p.get("csls", 0) >= 0.025 and p.get("f") is not None and 0 <= int(p["f"]) < len(frame_files):
        fpath = frames_dir / frame_files[int(p["f"])]
        if fpath.exists():
            try:
                im = Image.open(fpath).convert("RGB")
                pad = int(p.get("s", 224) * 0.2)
                x0 = max(0, int(p.get("x", 0)) - pad)
                y0 = max(0, int(p.get("y", 0)) - pad)
                x1 = min(im.width, int(p.get("x", 0) + p.get("s", 224)) + pad)
                y1 = min(im.height, int(p.get("y", 0) + p.get("s", 224)) + pad)
                return im.crop((x0, y0, x1, y1)).resize((tile_size, tile_size), Image.Resampling.BILINEAR), "video_frame"
            except Exception:
                pass
    return render_object_crop(w, rgb, op, sc, walk_pts, obj, W=tile_size, H=tile_size), "splat_3d"


def classify_batch(client, batch_idx, batch_items, vocab_classes, floor_y):
    tile = 192
    cols = 6
    rows_n = (len(batch_items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows_n * tile), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)

    meta_rows = []
    for k, (obj, entry, crop_im, crop_src) in enumerate(batch_items):
        gx, gy = (k % cols) * tile, (k // cols) * tile
        sheet.paste(crop_im, (gx, gy))
        draw.rectangle([gx, gy, gx + tile - 1, gy + tile - 1], outline=(70, 80, 95), width=1)
        draw.rectangle([gx + 2, gy + 2, gx + 78, gy + 20], fill=(10, 12, 16))
        draw.text((gx + 6, gy + 4), obj["id"], fill=(110, 231, 183))

        s = obj["size"]
        base_h = round(float(obj["center"][1] - s[1] / 2 - floor_y), 2)
        prior = (entry or {}).get("final") or (entry or {}).get("render") or {}
        p_id = prior.get("id", obj["cls"])
        vc_map = {c["id"]: c["name"] for c in vocab_classes}
        meta_rows.append({
            "object_id": obj["id"],
            "tile_row": k // cols + 1,
            "tile_col": k % cols + 1,
            "crop_source": crop_src,
            "size_m": [round(float(x), 2) for x in s],
            "elevation_above_floor_m": base_h,
            "rgb": [round(float(x), 2) for x in obj["color"]],
            "geometry_vision_prior": {
                "id": p_id,
                "label": vc_map.get(p_id, prior.get("label", obj["label"])),
                "score": round(float(prior.get("score", 0.4)), 2),
            },
        })

    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=85)
    img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

    class_list = "\n".join(f'- "{c["id"]}": {c["name"]} (kind={c["kind"]})' for c in vocab_classes)
    prompt = (
        "You are an expert facility and boiler-room engineer inspecting a 3D digital twin of an equipment/boiler room.\n"
        "Attached is a grid of object crops from this room (each tile is labeled with its `object_id`, e.g. OBJ-000), "
        "along with each object's 3D bounding-box size `[W, H, D]` in meters, elevation above floor, RGB color, and `geometry_vision_prior`.\n\n"
        "IMPORTANT CONTEXT:\n"
        "1. This room contains BOTH industrial boiler-room equipment (expansion tanks, boilers, circulation pumps, electric motors, control valves, pressure gauges, sensors, electrical cabinets, insulated pipes, ventilation ducts) AND ordinary everyday / utility items actually kept in the room (a washing machine / dryer, cardboard boxes, plastic storage bins, an aluminium stepladder, chairs/stools, wheeled carts, ceiling lamps, walls/doors).\n"
        "2. Name every object honestly according to what it actually is: e.g., if `id` is `APPLIANCE`, label it `'Washing machine'` or `'Front-loading washing machine'`; if `BOX`, label it `'Cardboard storage box'` or `'Plastic storage bin'`; if `LADDER`, `'Aluminium step ladder'`; if `CHAIR`, `'Office chair'` or `'Work stool'`; if `CART`, `'Wheeled utility cart'` or `'Pallet jack'`; and for industrial equipment use accurate technical names (`'Diaphragm expansion vessel'`, `'Inline circulation pump'`, `'Motorized control valve'`, etc.).\n"
        "3. When `geometry_vision_prior.score >= 0.40`, keep `id` equal to `geometry_vision_prior.id` and write a specific, natural `label` (2-5 words) and `reason` matching that class and crop.\n"
        "4. For EVERY object, provide:\n"
        "   - `object_id`: exact ID (e.g. 'OBJ-000')\n"
        "   - `id`: canonical class ID from the allowed list below\n"
        "   - `label`: specific 2-5 word name (industrial name for equipment, everyday name for utility/clutter items)\n"
        "   - `score`: confidence 0.50..0.96\n"
        "   - `reason`: 1 concise sentence describing the object's visual & 3D physical features and location in the room.\n\n"
        f"Allowed `id` values:\n{class_list}\n\n"
        f"Objects in this grid:\n{json.dumps(meta_rows, ensure_ascii=False)}\n\n"
        "Return ONLY a JSON array of objects: "
        '[{"object_id": "OBJ-...", "id": "<CLASS_ID>", "label": "<Specific name>", "score": 0.88, "reason": "<1 sentence>"}]'
    )

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[img_part, prompt],
                config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                import time
                time.sleep(20 * (attempt + 1))
                continue
            raise
    parsed = json.loads(resp.text or "[]")
    if isinstance(parsed, dict):
        parsed = parsed.get("objects") or parsed.get("results") or [parsed]
    print(f"[gemini-batch {batch_idx:02d}] classified {len(parsed)}/{len(batch_items)} objects")
    return {item["object_id"]: item for item in parsed if isinstance(item, dict) and "object_id" in item}


def main():
    scene_path = ROOT / "web" / "scene.json"
    vl_path = ROOT / "web" / "vision_labels.json"
    vocab_path = ROOT / "web" / "vocab.json"
    frames_dir = ROOT / "web" / "frames"

    scene = json.loads(scene_path.read_text())
    vl = json.loads(vl_path.read_text())
    vocab = json.loads(vocab_path.read_text())
    vocab_by_id = {c["id"]: c for c in vocab["classes"]}
    frame_files = json.loads((frames_dir / "index.json").read_text())["files"] if (frames_dir / "index.json").exists() else []

    ply_path = ROOT / "data" / "equipment_room.ply"
    if not ply_path.exists():
        ply_path = ROOT / "web" / "boiler_room_clean.ply"
    print(f"[gemini] loading splats from {ply_path.name}...")
    w, rgb, op, sc = load_splat_cloud(ply_path)

    g = scene["grid"]
    walk_pts = []
    for j in range(g["h"]):
        for i in range(g["w"]):
            if g["walkable"][j * g["w"] + i] == "1":
                walk_pts.append([g["origin"][0] + (i + 0.5) * g["cell"], g["origin"][1] + (j + 0.5) * g["cell"]])
    walk_pts = np.array(walk_pts, dtype=np.float64)

    only_missing = "--only-missing" in sys.argv
    only_classes = None
    if "--classes" in sys.argv:
        idx = sys.argv.index("--classes")
        if idx + 1 < len(sys.argv):
            only_classes = set(sys.argv[idx + 1].split(","))
    objs = scene["objects"]
    if only_missing:
        objs = [o for o in objs if not vl["labels"].get(o["id"], {}).get("gemini")]
    if only_classes:
        objs = [o for o in objs if vl["labels"].get(o["id"], {}).get("final", {}).get("id") in only_classes]
    print(f"[gemini] rendering crops for {len(objs)} objects...")
    items = []
    for o in objs:
        entry = vl["labels"].get(o["id"], {})
        crop_im, crop_src = get_best_crop(o, entry, frames_dir, frame_files, w, rgb, op, sc, walk_pts)
        items.append((o, entry, crop_im, crop_src))

    batch_size = 24
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {
            pool.submit(classify_batch, client, idx, b, vocab["classes"], scene["floor_y"]): idx
            for idx, b in enumerate(batches)
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                results.update(fut.result())
            except Exception as e:
                print(f"[gemini-batch {idx:02d}] error: {e}")

    updated = 0
    for o in objs:
        oid = o["id"]
        entry = vl["labels"].setdefault(oid, {})
        prior = entry.get("final") or entry.get("render") or {}
        gem = results.get(oid)
        if not gem:
            continue
        cid = gem.get("id")
        if cid not in vocab_by_id:
            cid = prior.get("id", "PIPE")
        # Preserve high-confidence geometry+CLIP class ID when prior is >= 0.44 to avoid breaking calibrated 3D box classes,
        # while upgrading unconfident (< 0.44) boxes to Gemini's class ID and using Gemini's specific industrial label & reason everywhere.
        prior_id = prior.get("id")
        prior_sc = float(prior.get("score", 0))
        if prior_id in vocab_by_id and prior_sc >= 0.40:
            final_id = prior_id
        else:
            final_id = cid
        vc = vocab_by_id[final_id]
        gem_label = str(gem.get("label") or vc["name"]).strip()
        if final_id != cid:
            gem_label = vc["name"]
        score = round(max(float(gem.get("score", 0.78)), prior_sc, 0.55), 3)
        reason = str(gem.get("reason") or f"{vc['name']} in industrial boiler room").strip()
        res_obj = {
            "id": vc["id"],
            "label": gem_label,
            "canonical_label": vc["name"],
            "kind": vc["kind"],
            "asset_class": vc["asset_class"],
            "score": score,
            "reason": reason,
            "source": MODEL,
        }
        entry["gemini"] = res_obj
        entry["final"] = res_obj
        updated += 1

    vl["model"] = f"{vl.get('model', 'Xenova/clip-vit-base-patch16')} + {MODEL}"
    vl_path.write_text(json.dumps(vl))
    print(f"[gemini] updated {updated}/{len(objs)} objects in {vl_path}")


if __name__ == "__main__":
    main()

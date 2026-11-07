"""Local multi-view VLM labels: every 3D box is shown to Qwen3-VL from several aisle viewpoints (renders from
multiview_classify.py --render) plus the matched walk-through video tile, with its size, height and colour, and the
model picks one class of web/vocab.json. Same job as gemini_vision.py, but offline on the GPU and with 3-6 views
instead of one crop.

Usage:
  python pipeline/vlm_classify.py work/mv2 --eval                 # benchmark on tests/vision_eval.json (50 boxes)
  python pipeline/vlm_classify.py work/mv2                        # all boxes -> work/mv2/vlm_labels.json
  python pipeline/vlm_classify.py work/mv2 --apply web/vision_labels.json   # merge as entry["multiview"]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from multiview_classify import video_tile, view_boxes, walkable_points  # noqa: E402

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
TILE = 320
MAX_VIEWS = 6


def contact_sheet(images, labels, boxes=None):
    """Views side by side, each with a caption and the object's projected 3D box in yellow, one image for the VLM."""
    n = len(images)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TILE, rows * (TILE + 18)), (20, 20, 24))
    d = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(zip(images, labels)):
        x, y = (i % cols) * TILE, (i // cols) * (TILE + 18)
        sx = TILE / im.width
        sheet.paste(im.resize((TILE, TILE)), (x, y + 18))
        d.text((x + 4, y + 3), lab, fill=(255, 230, 120))
        if boxes and boxes[i]:
            x0, y0, x1, y1 = boxes[i]
            d.rectangle([x + x0 * sx, y + 18 + y0 * sx, x + x1 * sx, y + 18 + y1 * sx], outline=(255, 220, 0), width=2)
    return sheet


def describe(o, floor_y):
    s = o["size"]
    base = o["center"][1] - s[1] / 2 - floor_y
    c = [int(255 * v) for v in o.get("color", [0.5, 0.5, 0.5])]
    return (f"3D box size: {s[0]:.2f} m wide x {s[1]:.2f} m tall x {s[2]:.2f} m deep; "
            f"its bottom is {base:.2f} m above the floor; mean colour RGB {c}.")


def build_prompt(vocab, n_wide, n_zoom, has_video, desc, prior=None):
    classes = "\n".join(f"- {c['id']}: {c['name']} (e.g. {', '.join(c['prompts'][:3])})" for c in vocab["classes"])
    parts = []
    if n_wide:
        parts.append(f"{n_wide} wide view(s) for context")
    parts.append(f"{n_zoom} zoomed view(s)")
    if has_video:
        parts.append("a sharp frame from the walk-through video (last image, no rectangle)")
    hint = ""
    if prior:
        hint = ("An image-text model (often wrong, use only as a hint) ranked: "
                + ", ".join(f"{k} {v:.2f}" for k, v in prior) + ".\n")
    return (
        f"{vocab['room_note']}\n\n"
        f"The images are {', '.join(parts)}, all of ONE region of a blurry 3D scan of this room. "
        f"The YELLOW rectangle marks that region in every rendered view.\n"
        f"The region is often only a PART of a bigger thing: a patch of a tank's side, a section of pipe, one step of a "
        f"ladder, a piece of wall. Name the class of the WHOLE object the region belongs to, using the wide views to see it.\n"
        f"{desc}\n{hint}\n"
        f"Choose exactly one ID from this list:\n{classes}\n\n"
        f"Rules:\n"
        f"- Bare wall, floor, ceiling, door, window, a flat painted surface or an unrecognisable blur -> STRUCTURE. "
        f"Most small patches in this scan are STRUCTURE.\n"
        f"- SENSOR and GAUGE are rare: only a clearly visible small round instrument or a puck with a cable on a pump/motor.\n"
        f"- Judge by what you SEE. Use the size and height only to rule out impossible classes "
        f"(a 5 cm box is not a tank; a thing 1.8 m up is not standing on the floor).\n"
        f"- confidence: 0.9 only when the whole object is clearly recognisable in at least two views; "
        f"0.6 when fairly sure; 0.3-0.4 when guessing from a blur.\n"
        f'Answer with JSON only: {{"class": "<ID>", "confidence": <0..1>, "name": "<3-5 word name>", "reason": "<one sentence>"}}'
    )


class VLM:
    def __init__(self, name=MODEL, dev="cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(dev).eval()
        self.proc = AutoProcessor.from_pretrained(name)
        self.name = name

    def ask(self, image, text, max_new_tokens=96):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def parse(answer, vocab_by_id):
    m = re.search(r"\{.*\}", answer, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        j = {}
    cid = str(j.get("class", "")).upper().strip()
    if cid not in vocab_by_id:  # tolerate "a PUMP" / lowercase / name instead of id
        hit = [k for k in vocab_by_id if k in cid or vocab_by_id[k]["name"].lower() == str(j.get("class", "")).lower()]
        cid = hit[0] if hit else None
    if not cid:
        return None
    c = vocab_by_id[cid]
    try:
        conf = float(j.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return dict(id=cid, label=c["name"], kind=c["kind"], asset_class=c["asset_class"], score=round(min(max(conf, 0), 1), 3),
                name=str(j.get("name", ""))[:60], reason=str(j.get("reason", ""))[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--apply", type=Path)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--scene", type=Path, default=ROOT / "web" / "scene.json")
    ap.add_argument("--labels", type=Path, default=ROOT / "web" / "vision_labels.json")
    ap.add_argument("--sheets", action="store_true", help="also save the contact sheets to work/sheets/")
    ap.add_argument("--wide", type=Path, help="work dir of a fixed-FOV render (multiview_classify --render with VIEW_FOV) for context views")
    ap.add_argument("--wide-fov", type=float, default=64.0)
    ap.add_argument("--prior", type=Path, help="multiview_labels.json from multiview_classify --label: its top-3 is given as a hint")
    a = ap.parse_args()

    scene = json.loads(a.scene.read_text())
    vocab = json.loads((ROOT / "web" / "vocab.json").read_text())
    vocab_by_id = {c["id"]: c for c in vocab["classes"]}
    vl = json.loads(a.labels.read_text())
    index = json.loads((a.work / "views.json").read_text())
    frames_dir = ROOT / "web" / "frames"
    frame_files = json.loads((frames_dir / "index.json").read_text())["files"] if (frames_dir / "index.json").exists() else []
    out_path = a.work / ("vlm_eval.json" if a.eval else "vlm_labels.json")
    done = json.loads(out_path.read_text())["labels"] if out_path.exists() and not a.apply else {}

    if a.apply:
        mv = json.loads((a.work / "vlm_labels.json").read_text())
        target = json.loads(a.apply.read_text())
        if mv["scene_id"] != target["scene_id"]:
            sys.exit(f"scene_id mismatch: {mv['scene_id']} vs {target['scene_id']}")
        prior = json.loads(a.prior.read_text())["labels"] if a.prior else {}
        for oid, lab in mv["labels"].items():
            if oid in prior:
                lab["prior_top3"] = prior[oid].get("top3")  # SigLIP multi-view vote, a second independent opinion
            target["labels"].setdefault(oid, {})["multiview"] = lab
        tmp = a.apply.with_suffix(".tmp")
        tmp.write_text(json.dumps(target))
        tmp.replace(a.apply)
        print(f"[apply] multiview entry added to {len(mv['labels'])} boxes in {a.apply}")
        return

    gt = json.loads((ROOT / "tests" / "vision_eval.json").read_text())["labels"] if a.eval else None
    objs = [o for o in scene["objects"] if (gt is None or o["id"] in gt) and o["id"] not in done]
    print(f"[vlm] {len(objs)} boxes to label with {a.model} ({len(done)} already done)")
    walk = walkable_points(scene)
    wide_index = json.loads((a.wide / "views.json").read_text()) if a.wide else {}
    prior = json.loads(a.prior.read_text())["labels"] if a.prior else {}
    vlm = VLM(a.model)
    if a.sheets:
        (a.work / "sheets").mkdir(exist_ok=True)
    t0 = time.time()
    for n, o in enumerate(objs):
        oid = o["id"]
        wide = [f for f in wide_index.get(oid, []) if f.endswith(("_az+0.jpg", "_top.jpg"))][:2]
        files = index.get(oid, [])[:MAX_VIEWS - len(wide)]
        ims = [Image.open(a.wide / "views" / f).convert("RGB") for f in wide] + [Image.open(a.work / "views" / f).convert("RGB") for f in files]
        labs = [f"wide view {i + 1}" for i in range(len(wide))] + [f"zoom view {i + 1}" for i in range(len(files))]
        vbw = view_boxes(o, walk, wide, fov=a.wide_fov)
        vb = view_boxes(o, walk, files)
        boxes = [vbw.get(f) for f in wide] + [vb.get(f) for f in files]
        tile = video_tile(vl["labels"].get(oid), frames_dir, frame_files)
        if tile is not None:
            ims.append(tile)
            labs.append("video frame")
            boxes.append(None)
        if not ims:
            continue
        sheet = contact_sheet(ims, labs, boxes)
        if a.sheets:
            sheet.save(a.work / "sheets" / f"{oid}.jpg", quality=85)
        top3 = prior.get(oid, {}).get("top3")
        ans = vlm.ask(sheet, build_prompt(vocab, len(wide), len(files), tile is not None, describe(o, scene["floor_y"]), top3))
        lab = parse(ans, vocab_by_id)
        if lab is None:
            print(f"[vlm] {oid}: unparsable answer: {ans[:100]!r}")
            continue
        lab.update(source="multiview", model=a.model, views=len(ims), video=tile is not None)
        done[oid] = lab
        if (n + 1) % 10 == 0 or n + 1 == len(objs):
            out_path.write_text(json.dumps(dict(scene_id=scene["scene_id"], model=a.model, labels=done), indent=0))
            print(f"[vlm] {n + 1}/{len(objs)}  {(time.time() - t0) / (n + 1):.1f} s/box  last: {oid} -> {lab['id']} {lab['score']} ({lab['name']})")
    if gt:
        correct = sum(1 for oid, t in gt.items() if done.get(oid, {}).get("id") == t)
        print(f"[eval] {a.model}: {correct}/{len(gt)} on tests/vision_eval.json")
        for oid, t in gt.items():
            p = done.get(oid, {})
            if p.get("id") != t:
                print(f"   {oid}: truth {t:10s} pred {p.get('id')!s:10s} {p.get('score')} {p.get('name')!r}: {p.get('reason')}")
    else:
        conf = sum(v["score"] >= vocab["min_score"] for v in done.values())
        print(f"[vlm] {len(done)} boxes labelled, {conf} with confidence >= {vocab['min_score']} -> {out_path}")


if __name__ == "__main__":
    main()

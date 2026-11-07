# theMOLFARS — Industrial Metaverse field-service demo

[![tests](https://github.com/huziichuk/theMOLFARS/actions/workflows/tests.yml/badge.svg)](https://github.com/huziichuk/theMOLFARS/actions/workflows/tests.yml)

**Valmet × Since AI 2026 challenge:** *Semantic 3D Scene Understanding & Autonomous Field-Service Copilot.*

We turn a raw 3D Gaussian Splatting scan of Valmet's Messukylä boiler/equipment room into an interactive digital twin:

1. **Semantic extraction.** 414 objects get non-overlapping 3D boxes.
2. **Excel mapping.** The synthetic Excel asset pack is mapped onto those objects.
3. **Navigation.** A service avatar walks a collision-free A* route to the faulty device, highlighting assets within a proximity radius as it goes.
4. **Diagnosis.** On arrival, a grounded AI copilot diagnoses the fault, citing exact evidence IDs.
5. **Vision.** A client-side CLIP model, helped by the real walk-through video, relabels what each box actually is. Optionally, Google Gemini names each object (batch script or live buttons).

![The viewer: splat with 3D frames, BIM floor plan, proximity list and copilot](docs/viewer_screenshot.jpg)

| Brief criterion (weight) | What we deliver |
|---|---|
| Semantic extraction & 3D framing (30%) | Colour-aware voxel-graph segmentation gives **414 boxes** with NMS (no overlaps). CLIP vision labels run in the browser and are improved by matching to the real video; Gemini vision refines them (49/50 on our benchmark). Touching parts of one thing are grouped (**49 multi-part objects from 129 fragments**), 88 boxes get a yaw-oriented frame, and any object can be **isolated** (only its splats shown) or downloaded as its own `.ply`. **Scan-to-BIM:** the whole semantic layer exports as **IFC4** (IfcPump, IfcValve, … with the Excel record as property sets; 0 schema errors in ifcopenshell) plus a 2D **floor plan and section A-A** (SVG). An optional **multi-view pass on a GPU PC** renders every box from 3-6 viewpoints with a PyTorch splat rasterizer and lets SigLIP 2 and a local VLM (Qwen3-VL) confirm or contest each label. |
| Obstacle-aware navigation & proximity (25%) | Floor/obstacle occupancy grid from the splat, an A* route with clearance cost, live 1.3 m proximity radius, collision-checked first-person walk mode, and an **inspection tour** over all 13 non-healthy assets ranked by condition × criticality × uncertainty (Builder Question Q-04). |
| AI diagnostics, grounding (20%) | Deterministic relational retrieval from PostgreSQL (seeded from the Excel pack) feeds Google Gemini (`gemini-3.5-flash-lite`) or an offline evidence report. Every claim cites `[EVT-…] [WO-…] [CASE-…] [DOC-…] [DQ-…]`; confidence drops with data-quality flags and a weak 3D identity (Builder Question Q-06); one click drafts a work order (Q-05). The copilot **comments on every asset the avatar passes**, **checks every cited ID** against the retrieved evidence, and runs its own **telemetry anomaly detector** (AUC 0.91 against the pack's labels, Q-02). |
| Backend architecture (15%) | A modular Python pipeline (`pipeline/`), a FastAPI server over PostgreSQL (Docker Compose, SQLite fallback) implementing every route of the pack's `API_Contract` sheet, live telemetry ingest + MQTT bridge (Q-07), OpenUSD export (Q-09), a static three.js + Spark viewer, `docker compose up` for everything, and CI on every push. |
| Demo clarity (10%) | A first-visit intro (what this is, four steps, colours, keys; reopen with **?**), a grouped toolbar with one highlighted main action, a colour legend, plain-language mission card and one-click suggestions in the chat. One-click mission (Start → walk → arrive → diagnosis → work-order draft), Follow / Orbit / Walk cameras, real-photo asset cards, a scripted **🎬 Guided demo** with captions for the video, a noise-free **🧱 BIM view** from above, **navigation by text** (“go to the nearest degraded valve”, also in Ukrainian), 8 slides in `docs/`. |

## Quick start

**Fastest:** put `builder_pack.xlsx` into `data/` and run `docker compose up --build`, then open http://localhost:8000. The first start downloads the splat from Google Drive, builds the scene and seeds PostgreSQL.

Requirements without Docker: Python ≥ 3.11, ffmpeg on PATH (or `imageio-ffmpeg`), a WebGPU browser (Chrome or Edge).

```bash
pip install -r requirements.txt
```

The starter data is not in git (size and licence). Put these files in `data/`:

| File | Source | Name in `data/` |
|---|---|---|
| Intact splat | `Equipment Room.ply` (from the mentors); **downloaded automatically** from the team's Google Drive by `build_scene.py` / `server.py` when missing (`pipeline/fetch_data.py`, `SPLAT_URL` to override) | `equipment_room.ply` |
| Walk-through video | `Industrial Boiler Room (1).mp4` (from the starter zip) | `boiler_room_360.mp4` |
| Excel pack | `Valmet_SinceAI_2026_Synthetic_Industrial_Metaverse_Builder_Pack.xlsx` | `builder_pack.xlsx` |

```bash
python pipeline/build_scene.py data/equipment_room.ply data/builder_pack.xlsx web    # ~5 s
python pipeline/extract_frames.py data/boiler_room_360.mp4 web/frames 1             # 112 frames
docker compose up -d                                                                 # PostgreSQL 16 on :5432
python seed_db.py                                                                    # Excel pack -> 14 tables
python server.py                                                                     # FastAPI on http://localhost:8000
# vision labels ship in web/vision_labels.json; after a rebuild press "Re-run AI labels" once, then optionally map Excel by what vision saw:
python pipeline/build_scene.py data/equipment_room.ply data/builder_pack.xlsx web --vision web/vision_labels.json
# optional, needs GEMINI_API_KEY: Gemini names every box, then rebuild so the mapping uses it
python pipeline/gemini_vision.py && python pipeline/build_scene.py data/equipment_room.ply data/builder_pack.xlsx web
```

`build_scene.py` uses `<outdir>/vision_labels.json` automatically when it exists (the log says so), and writes refined labels back into it.

Optional: put `GEMINI_API_KEY=…` in `.env` (git-ignored, read by `server.py`) to get LLM answers. Without it the copilot returns a fully cited, templated evidence report. `DATABASE_URL` overrides the default `postgresql+psycopg://molfars:molfars@localhost:5432/molfars`.

**No Docker?** Skip the two database lines. When PostgreSQL (or psycopg) is not available and `DATABASE_URL` is not set, `server.py` seeds `work/builder_pack.sqlite` from the same Excel file and runs the same SQL on it, so the demo never loses its copilot.

### Using the viewer
On the first visit an intro explains the demo in four steps and offers **▶ Start mission**, **🎬 Guided demo** or **Explore on my own**; the **?** button opens it again. The scene starts in an Orbit overview of the whole room; a legend in the corner explains the frame colours (red fault, amber warning, green healthy, grey not in Excel). Every button has a tooltip.

The toolbar, left to right:
- **▶ Start mission** (highlighted): the avatar walks to the faulty pump P-005; the camera switches to Follow and the copilot diagnoses on arrival. The mission card under the toolbar says the fault, health and walk length in plain words.
- **🧭 Inspection tour**: walks to all non-healthy assets, urgent first; each stop shows its priority and why.
- **🎬 Guided demo**: a 6-step scripted walkthrough with captions (overview, floor plan, mission, diagnosis, isolation, tour), for recording the video in one take.
- **Camera: Follow / Orbit / Walk** (WASD, arrows): Walk mode collides with the nav grid.
- **Show: All objects**: frames every detected object. Otherwise only frames inside the proximity radius are shown, with labels on the 6 nearest (name only; the confidence is in the label's tooltip).
- **Show: 🧱 BIM view**: hides the splat and shows solid blocks (status colours for assets, class colours otherwise) on the walkable floor, from above.
- **Show: 🗺️ Mini-map**: the 2D floor plan with the route; **⛶ Enlarge** makes it bigger.
- **AI: Re-run AI labels**: runs the in-browser CLIP vision pass again (~5 min the first time; the saved labels load without it).
- **AI: ✨ Scan this view**: asks Gemini to name the objects in the current view and saves the labels on the server (needs `GEMINI_API_KEY`).
- **⬇ Export ▾**: 📐 floor plan + section A-A (SVG), 🏗️ BIM model (IFC4, for Revit, Bonsai/BlenderBIM, Solibri), 🌐 OpenUSD (`.usda` for Omniverse, Unity, Unreal, Blender), and 📂 Open another splat (any local `.ply/.spz/.splat`).

In the chat and the asset card:
- **Welcome suggestions**: one-click Start mission, Diagnose P-005, Go to the nearest degraded valve, Inspection tour and BIM view.
- **Chat navigation**: type “go to the nearest degraded valve”, “take me to P-005”, “find the washing machine”, “знайди драбину”; the avatar walks the shortest real route (A* in the browser). Questions without a movement verb still go to the copilot.
- **Selecting an asset**: click a label or a proximity row. The card shows the 3D render and the matched real video photo, and you can ask the copilot about that asset.
- **Card buttons**: ✨ Gemini Vision (object), 🔍 Inspect (copilot), 📝 Work-order draft, 🔎 Isolate (only that object's splats, to orbit around it; Esc returns) and ⬇ PLY (that object's splats as a standalone `.ply`).

## Optional: multi-view labels on a GPU PC (`multiview_classify.py` + `vlm_classify.py`)

This is the brief's "multi-view vision models lifted to 3D", done without camera registration: every 3D box is **rendered from 3-6 aisle viewpoints** by a small Gaussian-splat rasterizer in PyTorch (`pipeline/splat_render.py`, the same picture Spark draws in the browser), the sharp video tile is added where the browser's photo match found one, and two independent models vote: **SigLIP 2** zero-shot over the `vocab.json` prompts, and **Qwen3-VL-4B** (a local vision-language model) that sees all views with the box drawn on them, its size and height, and the SigLIP ranking as a hint. The result is merged into `web/vision_labels.json` as a second opinion (`entry["multiview"]`): agreement with the Gemini label **confirms** it (score 0.75), disagreement keeps the label but **caps the confidence and shows the alternative** in the tooltip (0.45 when SigLIP votes for the same alternative, 0.6 when only the VLM disagrees), and a weak single-view label is replaced. On this scan: **195 boxes confirmed, 97 contested by both models, 122 by the VLM alone**; no label was flipped, the benchmark stays 49/50. On the 50-crop benchmark the browser CLIP render alone gets 26/50, SigLIP 2 multi-view 31/50, the VLM 32/50, and every box where the VLM agreed with Gemini (31/50) was right, so agreement is a usable confidence.

Needs an x64 PC with an NVIDIA GPU (tested: RTX 3080 Laptop 16 GB, Linux; ~25 min end to end). Models download from Hugging Face (~11 GB).

1. Put `boiler_room_360.mp4` and `builder_pack.xlsx` into `data/` (the `.ply` downloads itself).
2. Python 3.11/3.12 venv with CUDA PyTorch first (pick your CUDA on [pytorch.org](https://pytorch.org/get-started/locally/)), then the rest:
   ```bash
   python -m venv .venv-gpu && source .venv-gpu/bin/activate      # Windows: .venv-gpu\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements-gpu.txt
   ```
3. Render, vote, merge, rebuild:
   ```bash
   python pipeline/multiview_classify.py work/mv2 --render                 # zoomed views, ~10 min
   python pipeline/multiview_classify.py work/mv --render --wide           # context views
   python pipeline/multiview_classify.py work/mv2 --label --model google/siglip2-base-patch16-384
   python pipeline/vlm_classify.py work/mv2 --wide work/mv --prior work/mv2/multiview_labels.json   # ~20 min
   python pipeline/vlm_classify.py work/mv2 --apply web/vision_labels.json --prior work/mv2/multiview_labels.json
   python pipeline/build_scene.py data/equipment_room.ply data/builder_pack.xlsx web   # re-fuses the finals into scene.json
   ```
   `--eval` on either script scores the models on `tests/vision_eval.json` first; `--sheets` saves the contact sheets the VLM saw.
4. Commit `web/vision_labels.json` and `web/scene.json`; the viewer needs no GPU to show them.

**`video_lift.py` (COLMAP + CLIP-L + OWLv2)** is the registration-based variant and is still there. On this walk-through video COLMAP splits the frames into several reconstructions and the automatic splat alignment fails (ICP fitness 0.06–0.10 at 2 and 4 fps), so it needs ≥ 4 hand-picked `--pairs` before its `web/video_labels.json` is usable; the viewer ignores that file for any other `scene_id`.

## Submission package (brief §07)

| Deliverable | Where |
|---|---|
| Working demo, runnable locally | Quick start above; `python server.py` → http://localhost:8000 |
| Source repository with README and setup | this repository |
| 5–8 presentation slides | [docs/theMOLFARS_slides.pptx](docs/theMOLFARS_slides.pptx) (8 slides, speaker notes) and [PDF](docs/theMOLFARS_slides.pdf) |
| Demo video (≤ 5 min) | to record from the running demo; the shot list is in [WORKFLOW §3](docs/WORKFLOW.md) |
| Technical note | [TECH_NOTE.md](TECH_NOTE.md) |

## Documentation
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): architecture, every algorithm step, data formats and the HTTP API.
- [docs/WORKFLOW.md](docs/WORKFLOW.md): the end-to-end workflow (data → pipeline → demo), the GPU option, a demo script, the decisions log and the tools we used.
- [TECH_NOTE.md](TECH_NOTE.md): models, libraries and versions.
- [docs/theMOLFARS_slides.pptx](docs/theMOLFARS_slides.pptx): the 8-slide deck (pipeline, extraction, navigation, mapping, copilot, architecture, demo).

## Repository layout

```
pipeline/
  build_scene.py      raw .ply -> clean splat, floor, nav grid, 3D boxes, Excel mapping, A* route -> web/scene.json
  extract_frames.py   walk-through video -> web/frames/*.jpg + index.json
  splat_render.py     (GPU PC) Gaussian-splat rasterizer in PyTorch: renders any box from any viewpoint
  multiview_classify.py (GPU PC) 3-6 views per box + video tile, SigLIP 2 / CLIP-L vote, benchmark
  vlm_classify.py     (GPU PC) Qwen3-VL label per box from the views -> vision_labels.json["multiview"]
  video_lift.py       (GPU PC) COLMAP + CLIP-L + OWLv2 registration-based variant -> web/video_labels.json
  gemini_vision.py    Gemini labels for every box from contact sheets of crops -> web/vision_labels.json
  fetch_data.py       downloads the splat from Google Drive when data/ has none
  export_usd.py       scene.json -> OpenUSD (.usda), also served at /api/scene/export.usda
  export_bim.py       scene.json -> IFC4 (Scan-to-BIM) and the SVG floor plan + section, /api/scene/export.ifc, /plan.svg
  mqtt_bridge.py      MQTT topic plant/<ASSET>/telemetry -> POST /api/telemetry (pip install paho-mqtt)
server.py             FastAPI: serves web/ + the pack's API_Contract routes, vision, copilot, work-order drafts
seed_db.py            loads the Excel pack into PostgreSQL (14 sheets -> tables + indexes)
docker-compose.yml    PostgreSQL 16 + the app (Dockerfile); .github/workflows: tests on every push
web/
  index.html          three.js + Spark viewer, mission, proximity UI, copilot chat, in-browser CLIP vision
  scene.json          pipeline output (objects, grid, route, mapping)
  floorplan.png       debug/slide image of grid + boxes + route
docs/                 documentation, figures, slides (.pptx + .pdf)
inputs/ml_segment/    boxes from the team's ml branch (Open3D segment.py), for build_scene.py --boxes
web/vocab.json        CLIP model, classes, prompts, templates (shared by the browser pass and video_lift.py)
web/prototype_tiles.json   hand-labelled video tiles used as visual class prototypes
web/vision_labels.json     saved result of the browser vision pass (written by POST /api/vision)
tests/                unit tests (python -m unittest discover tests)
requirements.txt      pipeline + server deps;  requirements-gpu.txt: option-2 deps
```

## Results (on `Equipment Room.ply`)

| Area | Result |
|---|---|
| Splats | 1,676,481, all valid; the viewer gets 621,845 surface splats without view-dependent colour as `.spz` (**9.4 MB** instead of 396 MB) |
| Scan-to-BIM (IFC4) | 414 elements (18 IfcPump, 15 IfcValve, 5 IfcElectricMotor, 9 IfcFlowInstrument, 2 IfcSensor, 44 IfcTank, 15 IfcElectricDistributionBoard, 70 IfcPipeSegment, …), Pset_ValmetAsset on the 30 mapped assets; schema-valid, geometry matches the frames to the millimetre |
| Room height | floor y −1.17, ceiling y 0.87 (2.04 scan units) |
| Nav grid | 97×83 cells of 10 cm; 2,276 walkable (+59% after measuring clearance to the real splats; the rest of the floor is clutter or gaps narrower than 0.44 m) |
| Segmentation | 414 framed objects |
| Excel mapping | all 30 assets on boxes vision reads as pump / motor / valve / gauge / sensor, placed by a size-and-class fit score; target ASSET-017 / P-005 on OBJ-115, a reachable box vision reads as a pump |
| Route | 8.9 m, 11 waypoints, along the equipment wall; ≥ 0.224 m from every obstacle splat |
| Inspection tour (Q-04) | 13 stops, 57.2 m: 4 urgent (M-006, P-005, P-002, CV-003), 7 medium, 2 routine; all reachable |
| Frames | 414 boxes = 49 multi-part objects (129 fragments) + 285 single ones; 88 yaw-oriented |
| Telemetry detector (Q-02) | AUC 0.91 separating the 13 non-healthy from the 17 healthy assets, without using their labels |
| Tests | 27 unit tests (6 need the Excel pack and skip in CI) |
| Vision, CLIP only (B/16, prompt ensemble) | 302 / 414 boxes labelled ≥ 0.4 |
| Vision, saved labels (CLIP + refinement rules + Gemini) | 401 boxes labelled by Gemini, 13 by rules; **49 / 50** on the benchmark (see the caveats in HOW_IT_WORKS §2.3) |
| Vision benchmark (50 hand-labelled crops) | old setup 30% → **66%** correct; 74% precision on the labels actually shown |
| Copilot | offline evidence report verified for all 30 assets; for P-005 the overall confidence is 0.43 (model 0.91, 9 data-quality flags, 3D identity 0.55); covered by `tests` (`PostgresBackend`, runs on PostgreSQL or the SQLite fallback) |
| Proximity along the route | 14 Excel assets enter the 1.3 m radius before the target, among them M-006 (attention required), P-001, M-004, M-005 (degraded), M-001, P-004, CV-007, M-003 (watch) and 6 healthy |

## Known limits
- **The Gemini path of the copilot needs `GEMINI_API_KEY`** and was not exercised while writing these docs. The offline evidence path is the fallback on any API error.
- **`pipeline/video_lift.py` runs end to end but its automatic alignment fails on this video** (see the GPU section); the committed multi-view labels come from `multiview_classify.py` + `vlm_classify.py`, which need no registration. The VLM's self-reported confidence is not calibrated (it says 0.9 for almost everything), so the fusion uses agreement between models, not that number.
- **Boxes come from geometry.** Some are fragments of walls or pipes.
- **The committed Gemini scores are inflated.** They were made when `gemini_vision.py` still stored at least 0.55, so every box shows a name. The script now stores Gemini's own score; re-run it (needs a key) to get honest scores and "Unidentified object" back. Part of `refine_vision` is coordinate rules for this room; they switch off automatically on any other scan (17 of 414 labels here depend on them).
- **The Gemini vision endpoints write `scene.json` and `vision_labels.json` on the server.** Commit them only on purpose; a fresh `build_scene.py` run recomputes `scene.json` from the labels.
- **The original starter `Industrial Boiler Room.ply` is two-thirds corrupted.** It is random bytes after vertex 502,483, so half the room is missing. Use `Equipment Room.ply`; the pipeline detects and trims the broken file.
- **Not seen live here:** the narration while walking, the tour and the 🎬 Guided demo steps after the start were checked by code paths and unit tests, but the browser pane on the dev laptop was hidden, so the animation itself did not run. The Docker image was not built on the dev laptop (no Docker engine running). The layout was checked at 1440×900; below ~1200 px the mini-map and the legend cover a noticeable part of the scene.
- **The detector is modest on P-005:** its fouling shows as a slow flow drift; the detector ranks it 14th of 30, and the report says so.
- **OpenUSD export** is checked for structure locally and opened with `usd-core` in CI only (no ARM wheel). The IFC export was validated with ifcopenshell 0.9 locally and is again in CI; it was not opened in Revit.
- **Text navigation** understands tags, ~13 object kinds and 6 conditions by keywords (English, Ukrainian); it is not an LLM, so free phrasing outside those words falls back to the copilot or a short help line.

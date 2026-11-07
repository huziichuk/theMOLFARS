# How it works

## System overview

```mermaid
flowchart LR
  subgraph Offline["Offline pipeline (Python)"]
    PLY[Equipment Room.ply<br/>3D Gaussian splat] --> BS[build_scene.py]
    XLSX[Builder Pack .xlsx] --> BS
    BS --> SJ[web/scene.json<br/>boxes, grid, route, mapping]
    BS --> CP[web/boiler_room_clean.ply]
    MP4[Walk-through video .mp4] --> EF[extract_frames.py] --> FR[web/frames/*.jpg]
    MP4 -. GPU PC .-> VL[video_lift.py<br/>COLMAP + CLIP-L + OWLv2] -.-> VJ[web/video_labels.json]
    FR --> GV[gemini_vision.py<br/>contact sheets -> Gemini] --> VLJ[web/vision_labels.json]
    VLJ --> BS
  end
  subgraph Server["server.py (FastAPI)"]
    XLSX --> SEED[seed_db.py] --> PG[(PostgreSQL 16<br/>14 tables)]
    PG --> RAG[Relational retrieval<br/>8 tables per asset]
    RAG --> LLM{GEMINI_API_KEY?}
    LLM -- yes --> CL[Gemini gemini-3.5-flash-lite]
    LLM -- no --> OFF[Offline cited report]
  end
  subgraph Browser["Viewer web/index.html"]
    SJ --> V[three.js + Spark splat renderer]
    CP --> V
    FR --> VIS[CLIP in transformers.js<br/>WebGPU]
    VJ --> V
    V --> MIS[Mission: A* walk, proximity radius]
    MIS -- arrival --> API[POST /api/copilot]
    VIS --> V
    V -- buttons --> GAPI[POST /api/vision/gemini*]
  end
  API --> RAG
  GAPI --> CL
```

The heavy, deterministic work runs offline in about 5 s. It turns the raw splat into a semantic layer (`scene.json`). The browser renders the splat with the semantic layer on top and runs the mission. It can also run a vision model locally. The server stays thin: static files, the copilot endpoint, and the Gemini vision endpoints that save labels.

---

## 1. Offline pipeline — `pipeline/build_scene.py`

Input: a binary little-endian 3DGS `.ply` (float32 properties `x y z scale_* rot_* f_dc_* opacity f_rest_*`) and the Excel pack.
Output: `web/scene.json`, `web/boiler_room_clean.ply`, `web/floorplan.png`.

### 1.1 Load and sanitize
- The header is parsed and the body is read as an `N × P` float32 matrix (`np.frombuffer`, zero-copy).
- **Sanity mask:** a row is valid if all its values are finite, |xyz| < 50, and log-scales are in (−20, 3).
- **Corrupted-tail detection.** The whole rest of the file is cut from the first invalid row only if fewer than 50% of the rows after it are valid **and** the file's last 1,000 rows are mostly invalid. A bad block in the middle of an otherwise good file loses only its own rows. `tests/test_build_scene.py` covers a clean file, a garbage tail and a mid-file bad block. The original starter file (`Industrial Boiler Room.ply`) is valid only up to vertex 502,483. After that it holds noise (entropy 7.92 bits/byte, no float structure). Splats are stored in octree order, so the missing part is a whole region of the room.
- **Slim splat for the viewer.** Only splats that cover surfaces (`max scale > 3 cm` and opacity ≥ 0.08) or are clearly visible (opacity ≥ 0.20) are kept, and the view-dependent colour (`f_rest_*`, 45 of 59 properties) is dropped. That gives 621,845 of 1,676,481 splats and 35 MB instead of 396 MB. Colours no longer change with the viewing angle. The full splat is still used for the grid, the boxes and the vision renders.

### 1.2 World frame, floor, ceiling
- The capture is y-down (COLMAP convention). World = rotate 180° about X: `(x, −y, −z)`. The viewer does the same with `mesh.rotation.x = π`.
- The floor and ceiling are the dominant peaks in the lower and upper thirds of a 2 cm histogram of opaque splat heights (opacity > 0.3).

### 1.3 Navigation grid (10 cm cells)
- **Observed floor** = cells with splats within ±12 cm of the floor height. A binary closing (4 iterations) and hole filling close gaps.
- **Obstacles** = cells with opaque splats between floor + 0.2 m and floor + 1.7 m (head height). A cell counts if it has ≥ 4 hits after a morphological opening, or ≥ 12 hits on its own.
- **Clearance:** a cell is blocked when any splat of an obstacle cell lies closer than the avatar radius (0.22 m) to the cell centre (k-d tree over the obstacle splats). Distances go to the real splats, not to whole 10 cm cells: the cell-based test made every gap 10–15 cm narrower than it is and cut the corridor along the equipment wall (a 0.9 m gap between an insulated pipe bend and a shelf) off from the rest of the room.
- **Result:** 2,276 walkable cells (1,431 with the first, dilation-based margin: +59%) for the same 0.22 m person radius. The rest of the floor is shelves, boxes, carts and machines, or gaps narrower than 0.44 m. The route keeps ≥ 0.224 m from every obstacle splat (checked every 2 cm). `tests` has a wall with a 0.6 m gap (passable) and a 0.3 m gap (not).
- **Walkable** = observed floor ∧ ¬inflated obstacle, keeping only the largest connected component.
- A clearance map (Euclidean distance transform) is kept for path costs.

### 1.4 Semantic extraction: 3D boxes
1. **Input splats:** opacity > 0.3, max scale < 0.25 m, between floor + 6 cm and ceiling − 20 cm.
2. **Voxelize** at 4 cm. Each voxel stores its splat count and mean colour (from the SH DC term: `0.5 + 0.282·f_dc`).
3. **Colour-aware connectivity:** a sparse graph links 26-neighbour voxels **only if their mean colours differ by < 0.10 per channel**. `scipy.sparse.csgraph.connected_components` gives the objects. This separates a red pump from the white insulated pipe it sits on and from the grey wall. Plain connectivity fused them into a few huge blobs: 46 boxes on the old file, versus 128 with colour, and 414 on the intact file.
4. **Box per component:** the 2nd–98th percentile extent, which drops stray splats. Specks (< 8 cm) and slabs (> 2.5 m: walls, ceiling) are discarded, as are components with fewer than 30 splats.
5. **NMS:** a box is dropped if more than 35% of the smaller box lies inside an already-kept, bigger box. As a result, frames never stack on top of each other, as the brief requires.
6. **Optional union with the ml branch (`--boxes inputs/ml_segment`):**
   - Boxes from the team's Open3D `segment.py` are converted into this frame through its `transform.json` (yaw boxes become axis-aligned) and NMS'd.
   - Those that overlap none of ours by more than 35% are added.
   - Measured: +77 boxes, of which 6 read as equipment, so it is off by default.
7. **Heuristic class** from geometry and colour:
   - saturated red, mounted > 0.4 m above the floor, 0.2–0.8 m in size → *circulation pump*;
   - saturated red on the floor → *storage bin* (never mapped to an asset);
   - tall and thin → *vertical pipe*; long and flat → *horizontal pipe*;
   - small and high → *valve positioner / instrument*; small and low → *local measurement*;
   - very small (< 18 cm) and 0.3–1.3 m above the floor → *vibration / temperature sensor*;
   - low and compact → *motor/drive*; large → *tank/cabinet*.
8. **Vision refinement (`refine_vision`, with `--vision` or when `<outdir>/vision_labels.json` exists).** For each box the render and photo labels are merged (photo wins on agreement, or when CSLS ≥ 0.03 and it is ≥ 0.12 more confident). Then about 30 geometry rules correct it:
   - generic physical rules, e.g. a cart can't float more than 0.35 m above the floor, a pump can't be taller than 1.05 m, a thin tubular shape is a pipe, a saturated-red pipe-mounted box is a pump;
   - **rules tied to this scan's coordinates**, e.g. "x > 3.7, z > 2.3, ~0.5 m tall → pump (east-wall manifold)" or "x > 3.5, −1.5 < z < 0.2 → ladder". These encode what we saw in this room, so they run **only when the box geometry is this scan's** (`scene_id` 9dd32119e0b1, `ROOM_RULES_SCENE`); on any other scan they switch off. Here they decide 17 of 414 labels.
   - A Gemini label for the same class keeps its specific name and reason.

   The refined label is written back into `vision_labels.json` (atomically; the log says so).

### 1.5 Excel mapping
- `Asset_Master` is joined with `Current_State`. The **worst health score becomes the mission target** (ASSET-017, P-005, process pump, fouling, health 44).
- The mapping is procedural, which the brief explicitly allows. Assets are placed **worst health first**, each on the free box with the best `asset_fit_score`:
  - +4 for the exact class, +1.5 for a compatible one (pump ↔ motor; valve ↔ transmitter ↔ vibration sensor), +2 if vision saw it, plus the vision score;
  - a size check per class (e.g. a pump box 0.20–0.85 m, a transmitter 0.08–0.30 m);
  - boxes vision reads as wall, pipe, duct, lamp, box, chair, ladder, cart or appliance are never used, and thin flakes (< 4 cm) or slabs (> 1.05 m) are pushed to the end.

  Assets of the same class are interchangeable to this score, so a rebuild can swap tags between boxes of one class (we saw 11 swaps); the classes and the target stay the same.
- **Target placement:** the target asset is moved onto a reachable object of exactly its class (≤ 1.3 m from an aisle, because pumps are wall-mounted). If none exists, a compatible class is used (≤ 1.0 m), then any equipment box (≤ 2.0 m). If even that is empty, the pipeline stops with a clear message. Among the candidates that really look like the device (its class, seen by vision, right size: fit score ≥ 8.5; all candidates if none does) it picks one far from the start, but not the very edge of the scan, so the walk passes more of the room. If no box received the target asset during mapping, the box gets the full Master + Current_State record, so health and status are always present.

### 1.6 Route
- The start is the walkable cell nearest the left edge of the scan. The goal is the walkable cell nearest the target box.
- **8-connected A\*** with no corner cutting. Step cost `c · (1 + 1.5 / (0.2 + clearance))`, so the path stays in the middle of aisles.
- **String-pulling:** the route is shortened by line of sight through walkable cells only (they already carry the avatar radius).
- Self-checks: the route must exist and every cell on it must be walkable (`assert`).

### 1.6a Inspection tour (Builder Question Q-04)
- **Priority** = `(1 − health/100) × criticality weight (A 1.0, B 0.75, C 0.5) × (1 + (1 − model_confidence))`: damage times importance, and an uncertain diagnosis is worth checking sooner. The same function ranks `/api/priorities`.
- **Order:** urgency level first (high → medium → routine, from `Current_State`), then inside each level the nearest next stop by real A* path length (greedy).
- Every non-healthy asset whose box is within 1.5 m of the floor is a stop; the rest are listed as unreachable. Result: 13 stops, 57.2 m, saved as `tour` in `scene.json` with each stop's path index.

### 1.6b Frames: groups and orientation (display layer)
- **Groups:** boxes of one vision class without an Excel asset that touch (≤ 3 cm) are union-found into one multi-part object (`GRP-…`): 49 groups from 129 fragments (pipe runs, shelves, walls). The viewer draws one frame per group.
- **Oriented frames:** a PCA of each object's floor footprint gives a yaw; when the rotated box covers < 80 % of the axis-aligned floor area it is stored as `obb` (88 boxes).
- Both are display-only: navigation, mapping and `scene_id` keep the raw axis-aligned boxes, so every saved vision label stays valid.

### 1.7 `scene.json` format
```jsonc
{
  "splat": "boiler_room_clean.ply", "frame": "rotate_x_180",
  "floor_y": -1.17, "ceiling_y": 0.87,
  "grid": { "origin": [x0, z0], "cell": 0.1, "w": 97, "h": 83, "walkable": "0101…" }, // row-major by z
  "objects": [{ "id": "OBJ-115", "cls": "PROCESS_PUMP", "label": "Circulation / centrifugal pump",
                "shape_label": "…",               // the geometry guess before vision
                "vision_id": "PUMP", "vision_score": 0.58, "vision_source": "gemini-3.5-flash-lite", "reason": "…",
                "center": [x, y, z], "size": [sx, sy, sz], "color": [r, g, b],
                "asset": { "asset_id": "ASSET-017", "tag": "P-005", "condition_state": "…", "health_score": 44, … } | null }],
  "target_object": "OBJ-115", "target_asset": "ASSET-017",
  "path": [[x, y, z], …],
  "scene_id": "9dd32119e0b1"   // SHA-1 of the box geometry; object IDs are valid only for these boxes
}
```

---

## 2. Viewer — `web/index.html`

This is a single static page with no build step. It loads three.js 0.180 and Spark 2.4 (a Gaussian-splat renderer for three.js) from a CDN through an import map.

- **Splat:** `SplatMesh({ url })`, rotated 180° about X to match the pipeline frame. *Open splat…* loads any local file through `fileBytes`.
- **Boxes:** `EdgesGeometry` line boxes coloured by status: green healthy, amber warning, pulsing red target, grey unmapped, cyan discovered from video. Labels are `CSS2DObject`s.
- **Mission:** the avatar (a capsule plus a proximity ring) moves along `path` at 0.45 m/s. On arrival it calls the copilot.
- **Proximity display:** each frame, only boxes within **1.3 m** are visible (or all of them with *All frames*). Only the **6 nearest** get text labels, with Excel assets first, so labels never pile up. The side panel lists the nearby assets with their status chips.
- **Cameras:** *Follow* (third person), *Orbit* (overview under the ceiling), *Walk* (first person). Walk mode moves with WASD/arrows and tests the next position against the walkable grid on each axis separately, so the user slides along walls instead of passing through them.

### 2.1 In-browser vision (CLIP, transformers.js)
Pressing **AI labels (CLIP)**:
1. **Loads `Xenova/clip-vit-base-patch16`** (set in `web/vocab.json`) through `@huggingface/transformers` 4.3.2 on **WebGPU**, with a WASM fallback.
   - It loads the vision and the text tower: `CLIPVisionModelWithProjection` and `CLIPTextModelWithProjection`, fp16 when the GPU has `shader-f16`, q8 otherwise.
   - The browser caches the weights after the first download.
2. **Renders each box:**
   - The camera is placed on the side of the nearest aisle cell (the view a technician gets), ≥ 0.8 m away because splats blur up close.
   - The main loop is paused. The view is rendered for 3 frames so Spark can re-sort the splats for the new viewpoint.
   - The projected box is cropped as a square with 80% context and scaled to 224².
3. **Classifies against `web/vocab.json`,** which `video_lift.py` also reads. It has 17 classes (14 at the time of the benchmark below; sensor, heat exchanger and duct were added later):
   - equipment: pump, motor, valve, gauge, vibration sensor, heat exchanger, cabinet, tank;
   - infrastructure: pipe, duct;
   - clutter: washing machine, chair, ladder, box, cart, lamp, wall/floor.

   Each class has 2–5 prompts. Its text embedding is the mean over all prompts × 4 templates (a *prompt ensemble*).
4. **Prototypes.** Each class also gets a visual prototype: the mean embedding of its 129 hand-labelled tiles from our walk-through video (`web/prototype_tiles.json`). The tiles are already embedded during video indexing.

   The score is `100 · (cos(text_c, v) + w · (cos(proto_c, v) − mean))` with `w = proto_weight` (0.25). A softmax over the classes gives the confidence.
5. **Decision:** confidence ≥ `min_score` (0.4) shows the class name. Below it the frame reads **"Unidentified object"** instead of a shape guess (the guess stays in the tooltip). Clutter frames that carry no Excel asset are faded.
6. **Saving.** The results go to `POST /api/vision` → `web/vision_labels.json`, keyed by `scene_id`. Every viewer loads them on start, so the labels are there without pressing the button again.

#### Benchmark (`tests/vision_eval.json`)
The test set is 50 render crops, hand-labelled by eye with 13 coarse classes; 22 ambiguous crops of the 72 sampled were left out. It measures accuracy at the object in the crop centre. With 50 samples, differences of ±7 points are noise.

| Variant | CLIP B/32 | **CLIP B/16** | CLIP L/14 |
|---|---|---|---|
| Old vocabulary (16 prompts, 1 template) | 30% | 58% | 52% |
| New vocabulary, 1 template | 52% | 64% | 62% |
| New vocabulary + 4-template ensemble | 58% | **66%** | 66% |
| Video prototypes only (kNN) | 50% | 46% | 50% |
| Ensemble + video prototypes (best weight) | 64% | 66% | 68% |

B/16 has about the same download size as B/32 and needs no prototypes to reach 66%. L/14 is ~4× larger and only marginally better. On B/16 at the 0.4 threshold, labels are shown for 38/50 crops with 74% precision.

What this measured:
- The main remaining errors are crops of large boxes whose framing shows a whole corner of the room, and confusions between pipe, valve and pump.
- The video prototypes add nothing measurable on B/16.

### 2.2 Photo match with the walk-through video
CLIP struggles on blurry splat renders, while the video shows the same equipment sharply.
1. `extract_frames.py` produces 112 frames at 1 fps.
2. Each frame is cut into **13 overlapping square tiles** (3 full-height, 10 half-height), and each tile is embedded with CLIP. The embeddings are computed once, then stored in the browser **Cache API**, keyed by a hash of the source video that `extract_frames.py` writes into `frames/index.json`.
3. Each box render is embedded and matched to tiles by **CSLS** (cross-domain similarity local scaling):
   `csls(o, t) = 2·cos(o, t) − r_o − r_t`, where `r` is the mean of the 10 best cosine similarities. With plain cosine, one blank-wall frame was the nearest neighbour of 105 of 414 renders (hubness). CSLS cut that to 37, and the remaining top pairs are visually correct.
4. Matches with CSLS ≥ 0.03 (calibrated by eye on match pairs) have their sharp tile classified. The photo label wins when it is more confident than the render label.
5. The asset card shows the render next to the matched real photo, cropped directly from the frame JPEG with CSS (no canvas needed).

The photo match reuses the tile embeddings, so no frame is decoded twice. Results are saved on the server (see 2.1). `scene_id` is a fingerprint of the box geometry, so it changes only when the boxes change. Labels from an older run are never applied to renumbered boxes.

### 2.3 Gemini vision
CLIP labels are the free, local baseline. With `GEMINI_API_KEY` there are three Gemini paths, all with `gemini-3.5-flash-lite` and the 17-class vocabulary from `vocab.json`. The prompt tells the model the room also holds everyday items (washing machine, boxes, ladder, chairs, carts).
1. **Batch, offline — `python pipeline/gemini_vision.py [--only-missing] [--classes PUMP,VALVE]`.**
   - Each box gets one 192 px crop: the matched video tile if CSLS ≥ 0.025, otherwise a CPU splat render from the nearest aisle.
   - 24 crops go on one labelled contact sheet; 4 sheets run in parallel. Each crop is sent with its box size, height above the floor, colour and the current label as a prior.
   - **The class is kept from the prior when the prior score is ≥ 0.40**; Gemini then only supplies the name and reason. The prior is the CLIP or rule label, never an earlier Gemini answer, so a re-run can still change Gemini's own classes (on the committed labels: 303 boxes keep their CLIP class, 111 are open to Gemini). The stored score is Gemini's own score for its class, or the prior's score when the prior class is kept. (The committed labels were made with an older floor of 0.55, see below.)
   - This was run for the committed labels: 401 of 414 boxes carry a Gemini label, the other 13 a refinement rule.
2. **One object — button *Gemini Vision (object)* in an object's chat card → `POST /api/vision/gemini`.** It sends the live render crop plus the matched video tile.
3. **The current view — button *Gemini scan view* → `POST /api/vision/gemini-scene`.** It sends the screenshot plus up to 30 projected candidate boxes; Gemini returns up to 14 detections, each matched to an object ID or null.

Paths 2 and 3 update `vision_labels.json` **and** `scene.json` on the server, so the next viewer sees the new labels at once.

**What the numbers mean.** On the 50-crop benchmark the saved labels are now **49/50** correct (CLIP render alone: 26/50 for the same files). Caveats:
- the committed labels were made with a score floor of 0.55 (since removed), which puts every Gemini label above `min_score`, so "Unidentified object" doesn't appear until `gemini_vision.py` is re-run;
- part of the gain comes from the scene-specific rules in 1.4 step 8 and from keeping the prior class, and we can't separate the two shares;
- the benchmark crops were known while the rules were written, so 49/50 is an upper bound for this room, not a generalisation result.

---

## 3. Multi-view lift (GPU PC) — `pipeline/video_lift.py`

This is the "2D foundation models lifted to 3D" path recommended by the brief. It needs an x64 machine with CUDA.

1. **SfM:** ffmpeg samples frames at 2 fps. Then pycolmap runs SIFT features → sequential matching → incremental mapping, which gives camera poses and a sparse point cloud.
2. **Alignment COLMAP → splat world:**
   - initial scale from the 5th–95th percentile extents;
   - FPFH features, then RANSAC with a scaled point-to-point estimate (Open3D);
   - ICP with scaling.
   The script writes `work/align_check.ply` (splat in grey, aligned COLMAP points in red) to check by eye. If the fitness is too low, the user passes ≥ 4 hand-picked point pairs and the Umeyama closed-form similarity is used instead.
3. **Multi-view labels:**
   - **Occlusion check:** a frame may vote for a box only if it observes at least 3 of the COLMAP points that lie inside the box. Boxes with fewer than 3 SfM points keep their browser label.
   - Each box's 8 corners are projected into every remaining frame. Views behind the camera, off-image or smaller than 40 px are rejected.
   - The 6 largest and closest views are kept and padded crops are classified with **CLIP ViT-L/14**.
   - The class probabilities are averaged over the views.
4. **New objects:**
   - **OWLv2** (`owlv2-base-patch16-ensemble`) detects the equipment vocabulary on about 120 frames.
   - Each detection is lifted to 3D from the COLMAP points observed inside its 2D box: median-centred, outliers trimmed, 10th–90th percentile box.
   - Detections that don't overlap an existing box by more than 30% are added.
5. **Output:** `web/video_labels.json` (`scene_id`, `labels` per object id, `extra_objects`, `world_from_colmap`). The viewer picks it up automatically, but ignores it (with a console warning) if its `scene_id` differs from the current `scene.json`. Multi-view labels take priority, and the new objects are drawn in cyan.

`python pipeline/video_lift.py --selftest` checks the Umeyama fit and the projection math without a GPU.

---

## 4. Backend and copilot — `server.py`, `seed_db.py`

### Data store
- `docker compose up -d` starts PostgreSQL 16 (`molfars-postgres`, port 5432; user, password and database default to `molfars`, overridable via `POSTGRES_*`).
- `seed_db.py` loads 14 sheets of the Excel pack into same-named lower-case tables. It replaces them on every run and adds indexes on `asset_id`, timestamps, `(asset_class, issue_code)` and `(doc_id, scope)`.
- `server.py` connects through SQLAlchemy + psycopg (`DATABASE_URL`, default `postgresql+psycopg://molfars:molfars@localhost:5432/molfars`) and reads `.env` with python-dotenv.
- **Fallback:** if that connection fails and `DATABASE_URL` is not set, `seed_db.seed()` writes the same 14 tables into `work/builder_pack.sqlite` (rebuilt when the Excel file is newer) and the server uses it. All queries are plain SQL that both databases run. With an explicit `DATABASE_URL` the server fails loudly instead.
- The pipeline (`build_scene.py`) still reads the Excel file directly for the asset mapping.

### HTTP API (FastAPI, uvicorn on 127.0.0.1)
| Method | Path | Body / response |
|---|---|---|
| GET | `/` and static files | served from `web/` |
| GET | `/api/assets` | all assets with their current state |
| GET | `/api/assets/{asset_id}` | the full evidence context for one asset (JSON); 404 if unknown |
| POST | `/api/vision` | `{scene_id, model, labels}` → writes `web/vision_labels.json` (scene_id must match `scene.json`, IDs `ABC-123`, ≤ 5 MB, atomic write) |
| POST | `/api/vision/gemini` | `{object_id, render_b64?, photo?}` → `{object_id, result}`; needs a key (503 without), 502 on a Gemini error |
| POST | `/api/vision/gemini-scene` | `{view_b64, candidates?}` → `{detections, updated}` |
| GET | `/api/assets/{asset_id}/telemetry` | every telemetry row of the asset, oldest first; 404 if unknown |
| GET | `/api/assets/{asset_id}/events` | its last 50 alarms/events |
| GET | `/api/assets/{asset_id}/documents` | RAG-ready knowledge: its asset card, its class guidance and the site-wide docs (`RAG_Corpus_Index` + `Knowledge_Docs` text) |
| GET | `/api/scene/anchors` | `Scene_Anchors` rows, each with the real 3D box it was mapped to (`object_id`, centre, size, vision class, identity confidence, `scene_id`) |
| POST | `/api/copilot` (also `/api/copilot/query`) | `{asset_id \| object_id, question?, history?}` → `{mode, answer}`. `object_id` is for boxes with no Excel asset: Gemini gets the vision label, size, height and the 4 nearest registered assets |
| GET | `/api/assets/{asset_id}/brief` | one cited status line (used while walking past) + its citation check |
| GET | `/api/assets/{asset_id}/anomalies` | the Q-02 detector: score, per-signal z_level / z_trend / flag, 1 h series |
| GET | `/api/priorities` | all 30 assets ranked by the Q-04 rule, with the reason |
| GET | `/api/objects/{object_id}/splat.ply` | only the splats inside that object's (or its group's) frame, for 🔎 Isolate; cached in `work/isolate/` |
| GET | `/api/scene/export.usda` | the semantic layer as OpenUSD (Q-09) |
| POST | `/api/telemetry` | `{asset_id, timestamp?, values, source?}` → appends one live reading (Q-07) |
| POST | `/api/work-orders/draft` | `{asset_id}` → a work-order **draft** in the `Maintenance_History` shape: priority from urgency, action, numbered steps (lockout first, cases, doc, data-quality checks), evidence IDs, confidence. Nothing is stored or sent |
| GET | `/health` | `{status: "ok"}` |

### Retrieval (deterministic and relational)
There are only 30 assets, so we use no vector store. For the asset ID the server runs parameterised SQL queries and collects:
- `Asset_Master`: identity and criticality;
- `Current_State`: health, issue code, confidence and urgency;
- `Telemetry_Sample_3d`: the trend of every numeric signal (first, last, min, max, mean);
- `Alarms_Events`: the latest high/critical/medium events;
- `Maintenance_History`: work orders;
- `Historical_Service_Cases`: cases with the same class and issue code;
- `Knowledge_Docs` and `Data_Quality` flags.

The result is about 11 k characters of JSON, and every row keeps its evidence ID.

### Telemetry anomaly detector (Builder Question Q-02)
Per signal of the 3-day telemetry: the first 2 days are the baseline (median, MAD); `z_level` = the last 6 h median against it, `z_trend` = the change of a line through 6 h medians over the window, both in robust baseline units. A signal is flagged at ≥ 2 units; the asset score is the RMS of the per-signal max over all signals. The pack's condition labels are never used, so they can grade it:

| Variant | AUC non-healthy vs healthy (30 assets) | P-005 rank |
|---|---|---|
| level only | 0.89 | 25 / 30 |
| **level + trend (shipped)** | **0.91** | 14 / 30 |

The report and the asset card show the score and the flagged signals; the card draws 3-day sparklines with the baseline shaded and the flagged tail in red (`/api/assets/{id}/anomalies`).

### Narration and citation check
- **Narration:** while the avatar walks, every Excel asset entering the radius gets one deterministic, cited line from `/api/assets/{id}/brief` (state, health, issue, strongest moving signal, last alarm ID). Tour stops get a stop line with the priority and why.
- **Citation check:** every answer (Gemini or offline) is scanned for `[ASSET|EVT|WO|CASE|DOC|DQ-…]` IDs and compared with the IDs in the retrieved context; the response carries `citations: {verified, unknown}`. The viewer strikes unknown IDs through in red and prints "✓ N cited IDs verified" or "⚠ … not in the evidence". The offline report and the narration lines verify 100 % by construction (tested).

### Live data and portability (Q-07, Q-09)
- `POST /api/telemetry {asset_id, timestamp?, values}` appends a reading to the same table the pack filled (known numeric columns only); the detector, copilot and sparklines see it at once. `pipeline/mqtt_bridge.py` forwards MQTT topics `plant/<ASSET>/telemetry`; an OPC UA client would post the same way.
- `GET /api/scene/export.usda` (and `pipeline/export_usd.py`): every frame as a scaled, yaw-rotated `Cube` with `valmet:*` attributes (object, vision class, asset, condition, health), plus the mission route and the tour as `BasisCurves`; Y-up, metres.

### Confidence (Builder Question Q-06)
Every context carries `confidence`: the pack's `model_confidence`, scaled down by each open data-quality flag of the asset (high ×0.85, medium ×0.93, low ×0.97), and the 3D identity (the vision score of the box the asset was mapped to). The overall value is the lower of the two: a diagnosis is only as sure as the data behind it and the certainty that this box is that device. For P-005: 0.91 → 0.43 after 9 flags; identity 0.55; overall 0.43. The offline report prints the breakdown; Gemini is told to state it and say why it is lower.

### Generation
- **With `GEMINI_API_KEY` (or `GOOGLE_API_KEY`):** Google Gemini `gemini-3.5-flash-lite` through the `google-genai` SDK, up to 2,000 output tokens; the evidence context goes into the system instruction and the chat history is passed as turns. The system prompt requires 4 sections: *what is wrong*, *why*, *safe next steps (lockout first)* and *data trust*. Every claim must cite IDs (`Builder_Questions` Q-03), and the model may answer only from the context.
- **Without credentials, or on any API error:** a templated report from the same context.
  - It lists telemetry signals that moved by more than 8%, and the last alarms and work orders.
  - Steps come from the matching historical cases and the knowledge doc, preceded by permit and lockout.
  - It ends with all data-quality flags.

### Setup automation
- **Splat from Google Drive:** `build_scene.py` and `server.py` call `pipeline/fetch_data.py` when the scan is missing; it downloads to a `.part` file, checks the `ply` magic (Drive returns an HTML page when a link is not public), then renames, so an interrupted download is retried.
- **`docker compose up --build`:** PostgreSQL + the app (`Dockerfile`, Python 3.12); the app seeds the database and, on first start, downloads the splat and builds the scene. Ports are bound to 127.0.0.1.
- **CI:** `.github/workflows/tests.yml` runs the unit tests on every push (with `usd-core`, so the USD export is opened by a real USD parser).

### Hardening
- **Every POST must be `Content-Type: application/json`, otherwise 415.** Any website open in the same browser could otherwise send a `text/plain` POST to `localhost:8000` (CSRF) and overwrite the labels or spend the Gemini quota; a JSON POST from another site needs a CORS preflight, which this server never approves.
- Request bodies are parsed defensively; `asset_id` or `object_id` is required. Questions are cut to 2,000 characters, and only the last 10 user/assistant messages, each cut to 4,000 characters, are passed through.
- An unknown asset returns 404 or 400 instead of an exception.
- Every POST body is limited to 5 MB. Images must be valid base64 data URLs. Gemini's `score` is clamped to 0–1 (a non-number becomes 0.85), and unknown or non-string object IDs are dropped.
- **Gemini calls run in a worker thread** (`asyncio.to_thread`), so a slow answer doesn't freeze the server. The label files are re-read and written in one step after the answer arrives, so parallel scans can't overwrite each other's labels.
- HTML, JSON and PLY are served with `Cache-Control: no-cache`: the browser revalidates every time, so changed labels arrive at once, while an unchanged splat comes back as a 304 instead of being downloaded again.
- The viewer inserts object names as text (`esc()`), never as HTML, because they come from label files and Gemini.
- PostgreSQL listens on `127.0.0.1` only (`docker-compose.yml`), because the default password is in the repo.
- One room description (`room_note` in `vocab.json`) is shared by every Gemini prompt, and one rule (`mapping_class` in `build_scene.py`) turns a vision label into the mapping class for both the pipeline and the server.

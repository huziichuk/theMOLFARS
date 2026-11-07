# Technical note: models, libraries, components

| Layer | Component | Version / note | Where |
|---|---|---|---|
| Input 3D | `Equipment Room.ply`, a 3D Gaussian Splatting scan (59 float props, 1,676,481 splats) | from the Valmet mentors | `data/` (not in git) |
| Input data | Synthetic Industrial Metaverse Builder Pack (.xlsx: 30 assets, telemetry, 1,000 events, 300 work orders, 50 cases, docs, data-quality flags) | Since AI 2026 | `data/` |
| Input video | `Industrial Boiler Room (1).mp4`, 1920×1080, 112 s | starter pack | `data/` |
| Pipeline | Python ≥ 3.11, NumPy, SciPy (`ndimage`, `sparse.csgraph`), pandas + openpyxl, Pillow | | `pipeline/build_scene.py` |
| Segmentation | Colour-aware 26-neighbour voxel graph + connected components + 3D NMS; geometric/colour classifier | no learned model | `build_scene.py` |
| Navigation | 10 cm occupancy grid, morphology, obstacle inflation, EDT clearance, 8-connected A*, line-of-sight smoothing | | `build_scene.py` |
| Frames | ffmpeg on PATH, or the binary from `imageio-ffmpeg` | 9.x | `pipeline/extract_frames.py` |
| Viewer | three.js | 0.180.0 (CDN import map) | `web/index.html` |
| Splat renderer | Spark (`@sparkjsdev/spark`) | 2.4.0 | `web/index.html` |
| In-browser vision | transformers.js (`@huggingface/transformers`) on WebGPU, WASM fallback | 4.3.2 | `web/index.html` |
| Vision model | CLIP ViT-B/16 (`Xenova/clip-vit-base-patch16`): vision + text towers, fp16 on WebGPU, q8 fallback; 17 classes × prompt ensemble (4 templates); picked on a 50-crop benchmark (B/32, B/16, L/14 tried) | ONNX | browser |
| Visual prototypes | 129 hand-labelled tiles from the walk-through video | | `web/prototype_tiles.json` |
| Photo matching | CLIP tile embeddings + CSLS hubness-corrected nearest neighbour; Cache API | | browser |
| Gemini vision | `gemini-3.5-flash-lite` multimodal: batch contact sheets (`gemini_vision.py`), one object and current view (`/api/vision/gemini*`); JSON output constrained to the `vocab.json` classes | needs `GEMINI_API_KEY` | `pipeline/gemini_vision.py`, `server.py` |
| Label refinement | `refine_vision`: render/photo fusion + geometry and room-specific rules; `asset_fit_score` for the Excel mapping | | `build_scene.py` |
| Telemetry detector | Robust level + trend z-scores (median/MAD baseline), RMS over signals; NumPy | AUC 0.91 on the pack | `server.py` (`detect_anomalies`) |
| Planning | Q-04 priority, urgency tiers + greedy nearest stop by A* length | | `build_scene.py` (`priority`, `plan_tour`) |
| Interop | OpenUSD text export (no USD library); MQTT bridge (`paho-mqtt`, optional) | Omniverse / Unity / Unreal / Blender | `pipeline/export_usd.py`, `pipeline/mqtt_bridge.py` |
| Scan-to-BIM | IFC4 STEP writer (no IFC library) + SVG plan / section; validated with ifcopenshell | Revit, Bonsai, Solibri, xeokit | `pipeline/export_bim.py` |
| Splat format | Niantic `.spz` v2 writer (NumPy + gzip) for the viewer; float PLY kept for crops | Spark loads `.spz` natively | `build_scene.py` (`write_spz`) |
| Packaging | Dockerfile (Python 3.12) + Docker Compose, GitHub Actions CI, Google Drive download (stdlib `urllib`) | | `Dockerfile`, `.github/workflows/tests.yml`, `pipeline/fetch_data.py` |
| Data store | PostgreSQL 16 (Docker Compose), seeded from the Excel pack by `seed_db.py` (14 tables, indexed); SQLite fallback seeded the same way when PostgreSQL is not reachable | `postgres:16-alpine` | `docker-compose.yml`, `seed_db.py`, `server.py` |
| Copilot retrieval | Deterministic relational retrieval, parameterised SQL via SQLAlchemy + psycopg 3, pandas | | `server.py` |
| LLM | Google Gemini `gemini-3.5-flash-lite` via the `google-genai` SDK; offline evidence template without a key or on API errors | | `server.py` |
| Server | FastAPI + uvicorn, static files via `StaticFiles`; config from `.env` (python-dotenv) | | `server.py` |
| Optional multi-view lift | pycolmap (SfM), Open3D (FPFH, RANSAC, ICP with scaling), PyTorch CUDA, Hugging Face `transformers`: CLIP ViT-L/14, OWLv2 base-patch16-ensemble | x64 + NVIDIA GPU | `pipeline/video_lift.py` |

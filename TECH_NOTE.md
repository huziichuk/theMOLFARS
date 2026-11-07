# Technical note: models, libraries, components

| Layer | Component | Version / note | Where |
|---|---|---|---|
| Input 3D | `Equipment Room.ply`, a 3D Gaussian Splatting scan (59 float props, 1,676,481 splats) | from the Valmet mentors | `data/` (not in git) |
| Input data | Synthetic Industrial Metaverse Builder Pack (.xlsx: 30 assets, telemetry, 1,000 events, 300 work orders, 50 cases, docs, data-quality flags) | Since AI 2026 | `data/` |
| Input video | `Industrial Boiler Room (1).mp4`, 1920×1080, 112 s | starter pack | `data/` |
| Pipeline | Python ≥ 3.11, NumPy, SciPy (`ndimage`, `sparse.csgraph`), pandas + openpyxl, Pillow | | `pipeline/build_scene.py` |
| Segmentation | Colour-aware 26-neighbour voxel graph + connected components + 3D NMS; geometric/colour classifier | no learned model | `build_scene.py` |
| Navigation | 10 cm occupancy grid, morphology, obstacle inflation, EDT clearance, 8-connected A*, line-of-sight smoothing | | `build_scene.py` |
| Frames | ffmpeg | 9.x | `pipeline/extract_frames.py` |
| Viewer | three.js | 0.180.0 (CDN import map) | `web/index.html` |
| Splat renderer | Spark (`@sparkjsdev/spark`) | 2.4.0 | `web/index.html` |
| In-browser vision | transformers.js (`@huggingface/transformers`) on WebGPU, WASM fallback | 4.3.2 | `web/index.html` |
| Vision model | CLIP ViT-B/16 (`Xenova/clip-vit-base-patch16`): vision + text towers, fp16 on WebGPU, q8 fallback; 14 classes × prompt ensemble (4 templates); picked on a 50-crop benchmark (B/32, B/16, L/14 tried) | ONNX | browser |
| Visual prototypes | 129 hand-labelled tiles from the walk-through video | | `web/prototype_tiles.json` |
| Photo matching | CLIP tile embeddings + CSLS hubness-corrected nearest neighbour; Cache API | | browser |
| Copilot retrieval | Deterministic relational retrieval over 8 Excel sheets | pandas | `server.py` |
| LLM | Anthropic Claude `claude-opus-5-5` via the `anthropic` Python SDK (low effort, server-side refusal fallback) | SDK 2.6.0 | `server.py` |
| Server | Python stdlib `http.server` (`ThreadingHTTPServer`) | | `server.py` |
| Optional multi-view lift | pycolmap (SfM), Open3D (FPFH, RANSAC, ICP with scaling), PyTorch CUDA, Hugging Face `transformers`: CLIP ViT-L/14, OWLv2 base-patch16-ensemble | x64 + NVIDIA GPU | `pipeline/video_lift.py` |

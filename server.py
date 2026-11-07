"""FastAPI server: serves web/ and the grounded RAG copilot over PostgreSQL.

    python server.py            -> http://localhost:8000
Copilot uses Google Gemini 3.5 Flash Lite when GEMINI_API_KEY is available,
otherwise a deterministic evidence-only report built from the same context.
"""
import asyncio
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from PIL import Image
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent / "pipeline"))
from build_scene import mapping_class  # noqa: E402

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DB_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://molfars:molfars@localhost:5432/molfars")
MODEL = "gemini-3.5-flash-lite"
MAX_BODY = 5_000_000
ROOM_NOTE = json.loads((ROOT / "web" / "vocab.json").read_text())["room_note"]  # one description of the room for every Gemini prompt



def make_engine():
    """PostgreSQL when it is up. Without an explicit DATABASE_URL, a missing Docker or psycopg must not kill the demo:
    fall back to SQLite seeded from the same Excel pack (same tables, same SQL)."""
    try:
        eng = create_engine(DB_URL, pool_pre_ping=True)
        with eng.connect():
            return eng
    except Exception as e:
        if "DATABASE_URL" in os.environ:
            raise
        print(f"[db] PostgreSQL unavailable ({type(e).__name__}); using SQLite seeded from data/builder_pack.xlsx")
    import seed_db

    db = ROOT / "work" / "builder_pack.sqlite"
    if not db.exists() or db.stat().st_mtime < seed_db.DEFAULT_XLSX.stat().st_mtime:
        db.parent.mkdir(exist_ok=True)
        db.unlink(missing_ok=True)
        seed_db.seed(db_url=f"sqlite:///{db}")
    return create_engine(f"sqlite:///{db}")


engine = make_engine()


def records(df, n=None):
    df = df.drop(columns=["synthetic_record"], errors="ignore")
    if n:
        df = df.tail(n)
    return json.loads(df.to_json(orient="records", date_format="iso"))


def asset_context(asset_id: str) -> dict:
    """Deterministic relational retrieval from PostgreSQL: every fact about one asset, with evidence IDs."""
    with engine.connect() as conn:
        master = pd.read_sql(text("SELECT * FROM asset_master WHERE asset_id = :aid"), conn, params={"aid": asset_id})
        if master.empty:
            raise KeyError(asset_id)
        state = pd.read_sql(text("SELECT * FROM current_state WHERE asset_id = :aid"), conn, params={"aid": asset_id})
        cls, issue, doc = state.iloc[0][["asset_class", "issue_code", "knowledge_doc_id"]]
        tel = pd.read_sql(
            text("SELECT * FROM telemetry_sample_3d WHERE asset_id = :aid ORDER BY timestamp"),
            conn,
            params={"aid": asset_id},
        )
        alarms = pd.read_sql(
            text("SELECT * FROM alarms_events WHERE asset_id = :aid ORDER BY timestamp"),
            conn,
            params={"aid": asset_id},
        )
        maint = pd.read_sql(
            text("SELECT * FROM maintenance_history WHERE asset_id = :aid ORDER BY opened_at"),
            conn,
            params={"aid": asset_id},
        )
        cases = pd.read_sql(
            text("SELECT * FROM historical_service_cases WHERE asset_class = :cls AND historical_issue_code = :issue"),
            conn,
            params={"cls": cls, "issue": issue},
        )
        if cases.empty:
            cases = pd.read_sql(
                text("SELECT * FROM historical_service_cases WHERE asset_class = :cls"),
                conn,
                params={"cls": cls},
            )
            cases_records = records(cases, 3)
        else:
            cases_records = records(cases)
        docs = pd.read_sql(
            text("SELECT * FROM knowledge_docs WHERE doc_id = :doc OR scope = :cls"),
            conn,
            params={"doc": doc, "cls": cls},
        )
        dq = pd.read_sql(
            text("SELECT * FROM data_quality WHERE asset_id = :aid"),
            conn,
            params={"aid": asset_id},
        )

    num = tel.select_dtypes("number").dropna(axis=1, how="all")
    trend = {
        c: {
            k: round(float(v), 2)
            for k, v in dict(first=s.iloc[0], last=s.iloc[-1], min=s.min(), max=s.max(), mean=s.mean()).items()
        }
        for c, s in ((c, num[c].dropna()) for c in num.columns)
        if len(s)
    }
    dq_records = records(dq)
    state_rec = records(state)[0]
    return dict(
        asset=records(master)[0],
        state=state_rec,
        telemetry_trend_3d=trend,
        alarms=records(alarms[alarms.severity.isin(["high", "critical", "medium"])], 10) or records(alarms, 10),
        maintenance=records(maint, 8),
        similar_cases=cases_records,
        docs=records(docs),
        data_quality=dq_records,
        confidence=trust(state_rec["model_confidence"], dq_records, scene_box(asset_id)),
    )


DQ_PENALTY = {"high": 0.15, "medium": 0.07, "low": 0.03}
URGENCY_PRIORITY = {"high": "high", "medium": "medium", "routine": "low"}  # Current_State.urgency -> Maintenance_History.priority


def scene_box(asset_id: str) -> dict | None:
    """The 3D box this asset was mapped onto (web/scene.json), or None."""
    scene = json.loads((ROOT / "web" / "scene.json").read_text())
    return next((o for o in scene["objects"] if (o.get("asset") or {}).get("asset_id") == asset_id), None)


def trust(model_confidence, dq: list, box: dict | None) -> dict:
    """Builder question Q-06: lower the confidence when sensor data or the 3D mapping is poor.
    Every open data-quality flag scales the model confidence down; the overall value is the weakest link
    of that and the 3D identity (how sure vision is that the box really is this kind of device)."""
    data = float(model_confidence)
    for q in dq:
        data *= 1 - DQ_PENALTY.get(q.get("severity"), 0.07)
    identity = float(box.get("vision_score", 0.5)) if box else 0.0
    return dict(
        model=round(float(model_confidence), 2),
        after_data_quality=round(data, 2),
        data_quality_flags=len(dq),
        identity_3d=round(identity, 2),
        object_id=box["id"] if box else None,
        overall=round(min(data, identity), 2),
    )


SYSTEM = """You are the Valmet field-service copilot inside a 3D digital twin of a boiler room.
Answer ONLY from the CONTEXT (synthetic builder pack in PostgreSQL). If something is not in it, say so.
Cite evidence IDs in square brackets: [ASSET-…], [EVT-…], [WO-…], [CASE-…], [DOC-…], [DQ-…].
On arrival give a compact report with 4 headed sections:
1. What is wrong — diagnosis, health score, model confidence, abnormal telemetry (first -> last).
2. Why — linked alarms and maintenance history.
3. What to do next — numbered, safe inspection steps (lockout/permit first) from cases and docs.
4. Data trust — data-quality flags the technician must verify; give `confidence.overall` from the CONTEXT and say why it is lower than the model confidence.
Keep it under 250 words. Answer in the language of the user's question."""


def gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def ask_gemini(client, ctx, history, question):
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM + "\n\nCONTEXT:\n" + json.dumps(ctx, ensure_ascii=False),
            max_output_tokens=2000,
        ),
    )
    return resp.text or "No response generated."


def offline_report(ctx, question):
    """Fallback without an LLM: the same evidence, templated. Every line carries its source ID."""
    a, st = ctx["asset"], ctx["state"]
    out = [
        f"**{a['tag']} [{a['asset_id']}] — {a['asset_class']}**, {a['area_name']} ({a['criticality']})",
        "",
        "**1. What is wrong**",
        f"- {st['issue_summary']} (`{st['issue_code']}`), condition {st['condition_state']}, "
        f"health {st['health_score']}/100, model confidence {st['model_confidence']}, urgency {st['urgency']}.",
    ]
    for k, v in ctx["telemetry_trend_3d"].items():
        if v.get("first") is not None and v["first"] != 0:
            ch = (v["last"] - v["first"]) / abs(v["first"]) * 100
            if abs(ch) > 8:
                out.append(f"- `{k}`: {v['first']} -> {v['last']} ({ch:+.0f}%, range {v['min']}–{v['max']})")
    out += ["", "**2. Why (evidence)**"]
    out += [
        f"- [{e['event_id']}] {e['event_type']} / {e['event_code']} ({e['severity']}) {e['timestamp'][:16]}"
        for e in ctx["alarms"][-5:]
    ] or ["- No alarms recorded."]
    out += [f"- [{w['work_order_id']}] {w['action']} → {w['result']} ({w['opened_at'][:10]})" for w in ctx["maintenance"][-3:]]
    out += ["", "**3. What to do next**", "1. Obtain permit, apply lockout/tagout, confirm asset identity on site."]
    for i, c in enumerate(ctx["similar_cases"][:2], 2):
        out.append(f"{i}. {c['recommended_next_check'].capitalize()} — {c['historical_interpretation']} [{c['case_id']}]")
    for d in ctx["docs"][:1]:
        out.append(f"{len(ctx['similar_cases'][:2]) + 2}. {d['content']} [{d['doc_id']}]")
    out += ["", "**4. Data trust**"]
    out += [
        f"- [{q['issue_id']}] {q['issue_type']} ({q['severity']}) in {q['source_sheet']} → {q['recommended_handling']}"
        for q in ctx["data_quality"]
    ] or ["- No data-quality flags."]
    cf = ctx["confidence"]
    where = f" (`[{cf['object_id']}]`)" if cf["object_id"] else " (not mapped to a 3D box)"
    out.append(
        f"- **Overall confidence {cf['overall']:.2f}**: model {cf['model']:.2f} -> {cf['after_data_quality']:.2f} after "
        f"{cf['data_quality_flags']} data-quality flag(s); 3D identity {cf['identity_3d']:.2f}{where}; the lower one counts."
    )
    if question and not question.startswith("ARRIVED"):
        out += ["", f"_Offline mode: free-form question “{question}” needs an LLM — set GEMINI_API_KEY. Evidence above._"]
    return "\n".join(out)


app = FastAPI(title="Valmet Industrial Metaverse API")


@app.get("/api/assets")
def list_assets():
    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                "SELECT m.*, s.health_score, s.condition_state, s.issue_code, s.issue_summary, s.urgency "
                "FROM asset_master m JOIN current_state s USING (asset_id) ORDER BY m.asset_id"
            ),
            conn,
        )
    return records(df)


@app.get("/api/assets/{asset_id}")
def get_asset(asset_id: str):
    try:
        return asset_context(asset_id)
    except KeyError:
        return JSONResponse(status_code=404, content={"error": "unknown asset"})


def asset_rows(sql: str, asset_id: str, n: int | None = None):
    """Rows of one asset's table, oldest first; 404 for an unknown asset (API_Contract routes)."""
    with engine.connect() as conn:
        if pd.read_sql(text("SELECT 1 FROM asset_master WHERE asset_id = :aid"), conn, params={"aid": asset_id}).empty:
            return JSONResponse(status_code=404, content={"error": "unknown asset"})
        return records(pd.read_sql(text(sql), conn, params={"aid": asset_id}), n)


@app.get("/api/assets/{asset_id}/telemetry")
def asset_telemetry(asset_id: str):
    return asset_rows("SELECT * FROM telemetry_sample_3d WHERE asset_id = :aid ORDER BY timestamp", asset_id)


@app.get("/api/assets/{asset_id}/events")
def asset_events(asset_id: str):
    return asset_rows("SELECT * FROM alarms_events WHERE asset_id = :aid ORDER BY timestamp", asset_id, 50)


@app.get("/api/assets/{asset_id}/documents")
def asset_documents(asset_id: str):
    """RAG-ready knowledge: this asset's card, its class guidance and site-wide docs, with the text when the pack has it."""
    return asset_rows(
        "SELECT r.*, k.title, k.content FROM rag_corpus_index r LEFT JOIN knowledge_docs k ON k.doc_id = r.doc_id "
        "WHERE r.asset_id = :aid OR (r.asset_id IS NULL AND r.scope IN ('ALL', (SELECT asset_class FROM asset_master WHERE asset_id = :aid)))",
        asset_id,
    )


@app.get("/api/scene/anchors")
def scene_anchors():
    """Scene_Anchors from the pack, plus the real 3D box each asset was mapped to and how sure that identity is."""
    with engine.connect() as conn:
        anchors = records(pd.read_sql(text("SELECT * FROM scene_anchors ORDER BY asset_id"), conn))
    scene = json.loads((ROOT / "web" / "scene.json").read_text())
    boxes = {o["asset"]["asset_id"]: o for o in scene["objects"] if o.get("asset")}
    for a in anchors:
        o = boxes.get(a["asset_id"])
        a["scene_object"] = o and dict(object_id=o["id"], center_m=o["center"], size_m=o["size"], vision_id=o.get("vision_id"),
                                       identity_confidence=o.get("vision_score"), scene_id=scene["scene_id"])
        if o:
            a["mapping_status"] = "mapped_to_scan_box"
    return anchors


@app.post("/api/work-orders/draft")
async def work_order_draft(request: Request):
    """Q-05: 3D object -> evidence -> service case -> work-order DRAFT. Nothing is written or sent; no control actions."""
    try:
        ctx = asset_context(str((await read_json_body(request))["asset_id"]))
    except (KeyError, ValueError, TypeError):
        return JSONResponse(status_code=400, content={"error": "known asset_id required"})
    a, st = ctx["asset"], ctx["state"]
    steps = ["Obtain permit, apply lockout/tagout, confirm asset identity on site (QR/NFC)."]
    steps += [f"{c['recommended_next_check'].capitalize()} [{c['case_id']}]" for c in ctx["similar_cases"][:2]]
    steps += [f"Follow [{d['doc_id']}]" for d in ctx["docs"][:1]]
    steps += [f"Verify data-quality flag [{q['issue_id']}] {q['issue_type']}" for q in ctx["data_quality"][:3]]
    return dict(
        status="draft_not_submitted",
        work_order_id=None,  # the maintenance system assigns it when a person submits the draft
        asset_id=a["asset_id"], tag=a["tag"], area_name=a["area_name"],
        maintenance_type="inspection" if st["condition_state"] == "healthy" else "corrective",
        priority=URGENCY_PRIORITY.get(str(st["urgency"]).lower(), "medium"),
        action=st["recommended_next_check"],
        issue_code=st["issue_code"], issue_summary=st["issue_summary"],
        role="field_technician",
        knowledge_doc_id=st["knowledge_doc_id"],
        steps=steps,
        evidence=[e["event_id"] for e in ctx["alarms"][-5:]] + [w["work_order_id"] for w in ctx["maintenance"][-3:]]
        + [c["case_id"] for c in ctx["similar_cases"][:2]] + [q["issue_id"] for q in ctx["data_quality"]],
        confidence=ctx["confidence"],
    )


@app.post("/api/vision")
async def save_vision(request: Request):
    """Store the browser's vision pass so every viewer (and build_scene.py --vision) gets the same labels."""
    body = await request.body()
    if len(body) > MAX_BODY:
        return JSONResponse(status_code=413, content={"error": "too large"})
    try:
        req = json.loads(body)
        if not isinstance(req, dict):
            raise TypeError
        scene_id = json.loads((ROOT / "web" / "scene.json").read_text())["scene_id"]
        labels = req["labels"]
        ok = (
            req.get("scene_id") == scene_id
            and isinstance(labels, dict)
            and all(isinstance(k, str) and re.fullmatch(r"[A-Z]{3}-\d{3}", k) and isinstance(v, dict) for k, v in labels.items())
        )
    except (KeyError, ValueError, TypeError, OSError):
        ok = False
    if not ok:
        return JSONResponse(status_code=400, content={"error": "labels must be {OBJ-123: {...}} for the current scene_id"})
    out = ROOT / "web" / "vision_labels.json"
    if out.exists():
        try:
            prev = json.loads(out.read_text()).get("labels", {})
            for k, v in labels.items():
                if k in prev and prev[k].get("gemini") and "gemini" not in v:
                    v["gemini"] = prev[k]["gemini"]
                    v["final"] = prev[k]["gemini"]
        except Exception:
            pass
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"scene_id": scene_id, "model": str(req.get("model", "")), "labels": labels}))
    tmp.replace(out)
    return {"saved": len(labels)}


def crop_video_tile(photo: dict) -> bytes | None:
    try:
        idx_path = ROOT / "web" / "frames" / "index.json"
        if not idx_path.exists():
            return None
        files = json.loads(idx_path.read_text())["files"]
        f, x, y, s = int(photo["f"]), int(photo["x"]), int(photo["y"]), int(photo["s"])
        img = Image.open(ROOT / "web" / "frames" / files[f])
        crop = img.crop((x, y, x + s, y + s)).resize((384, 384))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return None


def vision_result(c: dict, d: dict, default_reason: str = "") -> dict:
    """One Gemini answer -> the label record stored in vision_labels.json (score clamped, text trimmed)."""
    try:
        score = min(1.0, max(0.0, float(d.get("score", 0.85))))
    except (TypeError, ValueError):
        score = 0.85
    return dict(
        id=c["id"],
        label=str(d.get("label") or c["name"]).strip()[:80],
        canonical_label=c["name"],
        kind=c["kind"],
        asset_class=c["asset_class"],
        score=round(score, 3),
        reason=str(d.get("reason") or default_reason).strip()[:240],
        source=MODEL,
    )


def save_gemini_labels(updates: dict) -> None:
    """Merge {OBJ-id: result} into vision_labels.json and scene.json.
    Re-reads both files and never awaits, so concurrent requests can't overwrite each other's labels."""
    scene_path, vlabels_path = ROOT / "web" / "scene.json", ROOT / "web" / "vision_labels.json"
    scene = json.loads(scene_path.read_text())
    vdata = json.loads(vlabels_path.read_text()) if vlabels_path.exists() else {"model": MODEL, "labels": {}}
    objs_by_id = {o["id"]: o for o in scene["objects"]}
    for oid, res in updates.items():
        entry = vdata.setdefault("labels", {}).setdefault(oid, {})
        entry["gemini"] = entry["final"] = res
        obj = objs_by_id[oid]
        obj["cls"] = mapping_class(res)
        obj.update(label=res["label"], reason=res["reason"], vision_source=MODEL, vision_id=res["id"], vision_score=res["score"])
    vdata["scene_id"] = scene["scene_id"]
    for path, data in ((vlabels_path, vdata), (scene_path, scene)):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)


async def read_json_body(request: Request) -> dict:
    body = await request.body()
    if len(body) > MAX_BODY:
        raise ValueError("too large")
    req = json.loads(body or b"{}")
    if not isinstance(req, dict):
        raise TypeError
    return req


def decode_data_url(s) -> bytes:
    if not isinstance(s, str) or "," not in s:
        raise ValueError("expected a data URL")
    return base64.b64decode(s.split(",", 1)[1], validate=True)


@app.post("/api/vision/gemini")
async def vision_gemini(request: Request):
    """Classify a 3D object using Gemini 3.5 Flash Lite multimodal vision (3D splat render + video crop + 3D box geometry)."""
    try:
        req = await read_json_body(request)
        oid = str(req["object_id"])
    except (KeyError, ValueError, TypeError):
        return JSONResponse(status_code=400, content={"error": "object_id required (body <= 5 MB)"})

    scene = json.loads((ROOT / "web" / "scene.json").read_text())
    vocab = json.loads((ROOT / "web" / "vocab.json").read_text())
    vocab_by_id = {c["id"]: c for c in vocab["classes"]}
    obj = next((o for o in scene["objects"] if o["id"] == oid), None)
    if not obj:
        return JSONResponse(status_code=404, content={"error": f"unknown object {oid}"})

    vlabels_path = ROOT / "web" / "vision_labels.json"
    entry = (json.loads(vlabels_path.read_text()) if vlabels_path.exists() else {}).get("labels", {}).get(oid, {})
    photo = req.get("photo") or entry.get("photo")

    parts = []
    try:
        raw = decode_data_url(req.get("render_b64"))
        parts.append(types.Part.from_bytes(data=raw, mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text="Image 1: 3D Gaussian Splatting render crop of the object."))
    except (ValueError, TypeError):
        pass
    if isinstance(photo, dict) and photo.get("csls", 0) >= 0.03:
        tile_bytes = crop_video_tile(photo)
        if tile_bytes:
            parts.append(types.Part.from_bytes(data=tile_bytes, mime_type="image/jpeg"))
            parts.append(types.Part.from_text(text="Image 2: Sharp walk-through video frame crop matched to this 3D object."))
    if not parts:
        return JSONResponse(status_code=400, content={"error": "render_b64 or matched video photo required for visual classification"})

    client = gemini_client()
    if not client:
        return JSONResponse(status_code=503, content={"error": "GEMINI_API_KEY not configured"})

    base_y = round(obj["center"][1] - obj["size"][1] / 2 - scene["floor_y"], 2)
    r_c, g_c, b_c = obj["color"]
    color_hint = "saturated red" if (r_c > 0.45 and g_c < 0.62 * r_c and b_c < 0.62 * r_c) else f"RGB({r_c:.2f}, {g_c:.2f}, {b_c:.2f})"
    class_desc = ", ".join(f"{c['id']} ({c['name']})" for c in vocab["classes"])
    prompt = (
        f"You are an expert facility and boiler-room engineer. Classify 3D object {oid} from the image(s) and 3D bounding box.\n"
        f"NOTE: {ROOM_NOTE} Name the object honestly according to what it is.\n"
        f"3D box size (W×H×D m): {obj['size']}, height above floor: {base_y} m, mean color: {color_hint}.\n"
        f"Allowed class IDs: {class_desc}.\n"
        'Return JSON: {"id": "<ALLOWED_CLASS_ID>", "label": "<Specific 2-5 word name>", "score": 0.0-1.0, "reason": "<1-sentence description>"}.'
    )
    parts.append(types.Part.from_text(text=prompt))
    try:
        resp = await asyncio.to_thread(  # off the event loop: the call takes seconds
            client.models.generate_content,
            model=MODEL,
            contents=parts,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1, max_output_tokens=300),
        )
        parsed = json.loads(resp.text)
        if isinstance(parsed, list):
            parsed = parsed[0] if (parsed and isinstance(parsed[0], dict)) else {}
        elif not isinstance(parsed, dict):
            parsed = {}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"Gemini vision failed: {e}"})
    raw_id = str(parsed.get("id") or "").upper()
    cid = raw_id if raw_id in vocab_by_id else ((entry.get("final") or {}).get("id") or "PIPE")
    res = vision_result(vocab_by_id[cid], parsed)
    save_gemini_labels({oid: res})
    return {"object_id": oid, "result": res}


@app.post("/api/vision/gemini-scene")
async def vision_gemini_scene(request: Request):
    """Run live Gemini 3.5 Flash Lite 2D object detection + 3D object classification on the current camera view."""
    try:
        req = await read_json_body(request)
        raw = decode_data_url(req["view_b64"])
        candidates = req.get("candidates") or []
        if not isinstance(candidates, list):
            raise TypeError
    except (KeyError, ValueError, TypeError):
        return JSONResponse(status_code=400, content={"error": "view_b64 data URL required (body <= 5 MB)"})

    client = gemini_client()
    if not client:
        return JSONResponse(status_code=503, content={"error": "GEMINI_API_KEY not configured"})

    scene = json.loads((ROOT / "web" / "scene.json").read_text())
    vocab = json.loads((ROOT / "web" / "vocab.json").read_text())
    vocab_by_id = {c["id"]: c for c in vocab["classes"]}
    known_ids = {o["id"] for o in scene["objects"]}

    img_part = types.Part.from_bytes(data=raw, mime_type="image/jpeg")
    class_desc = ", ".join(f"{c['id']} ({c['name']})" for c in vocab["classes"])
    cand_json = json.dumps(candidates[:30], ensure_ascii=False)
    prompt = (
        "You are an expert facility and boiler-room vision system inspecting a live 3D Gaussian Splatting camera view.\n"
        f"NOTE: {ROOM_NOTE} Name each visible object honestly.\n"
        f"Allowed class IDs: {class_desc}.\n"
        f"3D candidate boxes projected into this view (with normalized 2D boxes [ymin, xmin, ymax, xmax] in 0..1000 and 3D size in meters):\n{cand_json}\n\n"
        "Detect up to 14 distinct visible objects in the image, match each to the best `object_id` from the candidates list (or null if none matches), and return ONLY a JSON array:\n"
        '[{"object_id": "OBJ-...", "id": "<ALLOWED_CLASS_ID>", "label": "<Specific 2-5 word name>", '
        '"box_2d": [ymin, xmin, ymax, xmax], "score": 0.88, "reason": "<1-sentence description>"}]'
    )
    try:
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=[img_part, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
        )
        parsed = json.loads(resp.text or "[]")
        if isinstance(parsed, dict):
            parsed = parsed.get("detections") or parsed.get("objects") or [parsed]
        if not isinstance(parsed, list):
            parsed = []
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"Gemini scene scan failed: {e}"})

    detections = []
    updated = {}
    for d in parsed:
        if not isinstance(d, dict):
            continue
        raw_id = str(d.get("id") or "").upper()
        c = vocab_by_id[raw_id if raw_id in vocab_by_id else "PIPE"]
        res = vision_result(c, d, f"{c['name']} in boiler room")
        oid = d.get("object_id")
        oid = oid if isinstance(oid, str) and oid in known_ids else None
        box_2d = d.get("box_2d")
        if not (isinstance(box_2d, list) and len(box_2d) == 4):
            box_2d = None
        detections.append({**res, "object_id": oid, "box_2d": box_2d})
        if oid:
            updated[oid] = res
    if updated:
        save_gemini_labels(updated)
    return {"detections": detections, "updated": updated}


def object_scene_context(object_id: str) -> dict:
    scene = json.loads((ROOT / "web" / "scene.json").read_text())
    vlabels_path = ROOT / "web" / "vision_labels.json"
    vdata = json.loads(vlabels_path.read_text()) if vlabels_path.exists() else {"labels": {}}
    obj = next((o for o in scene["objects"] if o["id"] == object_id), None)
    if not obj:
        raise KeyError(object_id)
    entry = vdata.get("labels", {}).get(object_id, {})
    vis = entry.get("gemini") or entry.get("final") or {}
    c = obj["center"]
    s = obj["size"]
    base_y = round(c[1] - s[1] / 2 - scene["floor_y"], 2)
    nearby = []
    for other in scene["objects"]:
        if other["id"] == object_id or not other.get("asset"):
            continue
        oc = other["center"]
        dist = round(((c[0] - oc[0]) ** 2 + (c[1] - oc[1]) ** 2 + (c[2] - oc[2]) ** 2) ** 0.5, 2)
        nearby.append({
            "object_id": other["id"],
            "distance_m": dist,
            "tag": other["asset"].get("tag"),
            "asset_id": other["asset"].get("asset_id"),
            "asset_class": other["asset"].get("asset_class"),
            "condition_state": other["asset"].get("condition_state"),
            "issue_code": other["asset"].get("issue_code"),
        })
    nearby.sort(key=lambda x: x["distance_m"])
    return {
        "object_id": obj["id"],
        "class_id": vis.get("id") or obj["cls"],
        "specific_label": vis.get("label") or obj["label"],
        "canonical_label": vis.get("canonical_label") or obj.get("shape_label") or obj["label"],
        "kind": vis.get("kind", "equipment"),
        "confidence": vis.get("score", obj.get("vision_score", 0.75)),
        "vision_reason": vis.get("reason") or obj.get("reason", ""),
        "vision_source": vis.get("source") or obj.get("vision_source", MODEL),
        "dimensions_m": {"width_x": s[0], "height_y": s[1], "depth_z": s[2]},
        "elevation_above_floor_m": base_y,
        "center_xyz_m": c,
        "nearby_registered_assets": nearby[:4],
    }


def ask_gemini_scene_object(client: genai.Client, octx: dict, history: list, question: str) -> str:
    sys_prompt = (
        "You are a facility and boiler-room engineering copilot. "
        "The technician selected a 3D object detected by Gemini Vision that is NOT registered in the Excel telemetry database. "
        f"{ROOM_NOTE} "
        "Use the Gemini Vision classification, 3D physical measurements, elevation, and nearby registered assets to provide a practical assessment "
        "(what the object is, its visual/physical traits, whether it is industrial equipment or a room/utility item, safety/housekeeping or inspection notes, and relationship to nearby assets). "
        f"Reply in the language of the user's question.\n\nOBJECT CONTEXT:\n{json.dumps(octx, ensure_ascii=False)}"
    )
    contents = []
    for m in history:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    r = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=sys_prompt, max_output_tokens=700, temperature=0.2),
    )
    return r.text or ""


@app.post("/api/copilot")
@app.post("/api/copilot/query")  # the route name in the pack's API_Contract sheet
async def copilot(request: Request):
    try:
        req = await read_json_body(request)
    except (ValueError, TypeError):
        return JSONResponse(status_code=400, content={"error": "invalid json (body <= 5 MB)"})

    question = str(req.get("question") or "ARRIVED: run full diagnosis.")[:2000]
    raw_history = req.get("history")
    history = [
        {"role": m["role"], "content": m["content"][:4000]}  # a full report is ~2 k chars; the cap bounds the tokens sent
        for m in (raw_history if isinstance(raw_history, list) else [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ][-10:]

    if req.get("asset_id"):
        try:
            ctx = asset_context(str(req["asset_id"]))
        except KeyError:
            return JSONResponse(status_code=400, content={"error": "unknown asset"})
        client = gemini_client()
        if client:
            try:
                return {"mode": MODEL, "answer": await asyncio.to_thread(ask_gemini, client, ctx, history, question)}
            except Exception as e:
                print("[copilot] LLM unavailable:", type(e).__name__, e)
        return {"mode": "offline-evidence", "answer": offline_report(ctx, question)}

    if req.get("object_id"):
        try:
            octx = object_scene_context(str(req["object_id"]))
        except KeyError:
            return JSONResponse(status_code=400, content={"error": "unknown object"})
        client = gemini_client()
        if client:
            try:
                return {"mode": f"{MODEL} (scene-vision)", "answer": await asyncio.to_thread(ask_gemini_scene_object, client, octx, history, question)}
            except Exception as e:
                print("[copilot] Scene LLM unavailable:", type(e).__name__, e)
        d = octx["dimensions_m"]
        return {
            "mode": "offline-vision",
            "answer": (
                f"**{octx['specific_label']}** (`{octx['object_id']}`, class `{octx['class_id']}`, confidence {int(octx['confidence']*100)}%)\n"
                f"- **Gemini Vision**: {octx['vision_reason'] or octx['canonical_label']}\n"
                f"- **3D Dimensions**: {d['width_x']} × {d['height_y']} × {d['depth_z']} m (elevation +{octx['elevation_above_floor_m']} m)"
            ),
        }

    return JSONResponse(status_code=400, content={"error": "asset_id or object_id required"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.middleware("http")
async def json_posts_only(request: Request, call_next):
    """Any website can send a text/plain POST to localhost without asking (CSRF); application/json forces a CORS
    preflight, which this server never approves, so only our own page can call the API."""
    if request.method == "POST" and not request.headers.get("content-type", "").startswith("application/json"):
        return JSONResponse(status_code=415, content={"error": "Content-Type must be application/json"})
    return await call_next(request)


@app.middleware("http")
async def no_cache_html_json(request: Request, call_next):
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith(".html") or p.endswith(".json") or p.endswith(".ply"):
        # no-cache = revalidate every time: changed labels arrive at once, an unchanged 35 MB splat is a 304, not a re-download
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Valmet FastAPI demo on http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)

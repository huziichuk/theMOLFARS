"""FastAPI server: serves web/ and the grounded RAG copilot over PostgreSQL.

    python server.py            -> http://localhost:8000
Copilot uses Google Gemini 3.5 Flash Lite when GEMINI_API_KEY is available,
otherwise a deterministic evidence-only report built from the same context.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from sqlalchemy import create_engine, text

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DB_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://molfars:molfars@localhost:5432/molfars")
MODEL = "gemini-3.5-flash-lite"

engine = create_engine(DB_URL, pool_pre_ping=True)


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
    return dict(
        asset=records(master)[0],
        state=records(state)[0],
        telemetry_trend_3d=trend,
        alarms=records(alarms[alarms.severity.isin(["high", "critical", "medium"])], 10) or records(alarms, 10),
        maintenance=records(maint, 8),
        similar_cases=cases_records,
        docs=records(docs),
        data_quality=records(dq),
    )


SYSTEM = """You are the Valmet field-service copilot inside a 3D digital twin of a boiler room.
Answer ONLY from the CONTEXT (synthetic builder pack in PostgreSQL). If something is not in it, say so.
Cite evidence IDs in square brackets: [ASSET-…], [EVT-…], [WO-…], [CASE-…], [DOC-…], [DQ-…].
On arrival give a compact report with 4 headed sections:
1. What is wrong — diagnosis, health score, model confidence, abnormal telemetry (first -> last).
2. Why — linked alarms and maintenance history.
3. What to do next — numbered, safe inspection steps (lockout/permit first) from cases and docs.
4. Data trust — data-quality flags the technician must verify; overall confidence.
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


@app.post("/api/vision")
async def save_vision(request: Request):
    """Store the browser's vision pass so every viewer (and build_scene.py --vision) gets the same labels."""
    body = await request.body()
    if len(body) > 5_000_000:
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
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"scene_id": scene_id, "model": str(req.get("model", "")), "labels": labels}))
    tmp.replace(out)
    return {"saved": len(labels)}


@app.post("/api/copilot")
async def copilot(request: Request):
    body = await request.body()
    try:
        req = json.loads(body or b"{}")
        if not isinstance(req, dict):
            raise TypeError
        ctx = asset_context(str(req["asset_id"]))
    except (KeyError, ValueError, TypeError):
        return JSONResponse(status_code=400, content={"error": "asset_id required"})
    question = str(req.get("question") or "ARRIVED: run full diagnosis.")[:2000]
    raw_history = req.get("history")
    history = [
        m
        for m in (raw_history if isinstance(raw_history, list) else [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ][-10:]
    client = gemini_client()
    if client:
        try:
            return {"mode": MODEL, "answer": ask_gemini(client, ctx, history, question)}
        except Exception as e:
            print("[copilot] LLM unavailable:", type(e).__name__, e)
    return {"mode": "offline-evidence", "answer": offline_report(ctx, question)}


app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Valmet FastAPI demo on http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)

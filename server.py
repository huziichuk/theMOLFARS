"""Demo server: serves web/ and the grounded RAG copilot over the Excel builder pack.

    python server.py            -> http://localhost:8000
Copilot uses Claude when Anthropic credentials are available (ANTHROPIC_API_KEY or `ant auth login`),
otherwise a deterministic evidence-only report built from the same context.
"""
import json
import os
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
XLSX = ROOT / "data" / "builder_pack.xlsx"
SHEETS = ["Asset_Master", "Current_State", "Telemetry_Sample_3d", "Alarms_Events", "Maintenance_History",
          "Historical_Service_Cases", "Knowledge_Docs", "Data_Quality"]
MODEL = "claude-opus-5-5"


@lru_cache(1)
def sheets():
    return pd.read_excel(XLSX, sheet_name=SHEETS)


def records(df, n=None):
    df = df.drop(columns=["synthetic_record"], errors="ignore")
    if n:
        df = df.tail(n)
    return json.loads(df.to_json(orient="records", date_format="iso"))


def asset_context(asset_id):
    """Deterministic relational retrieval: every fact about one asset, with evidence IDs (30 assets -> no vector DB)."""
    s = sheets()
    master = s["Asset_Master"].query("asset_id == @asset_id")
    if master.empty:
        raise KeyError(asset_id)
    state = s["Current_State"].query("asset_id == @asset_id")
    cls, issue, doc = state.iloc[0][["asset_class", "issue_code", "knowledge_doc_id"]]
    tel = s["Telemetry_Sample_3d"].query("asset_id == @asset_id").sort_values("timestamp")
    num = tel.select_dtypes("number").dropna(axis=1, how="all")
    trend = {c: {k: round(float(v), 2) for k, v in dict(first=s.iloc[0], last=s.iloc[-1], min=s.min(), max=s.max(), mean=s.mean()).items()}
             for c, s in ((c, num[c].dropna()) for c in num.columns) if len(s)}
    alarms = s["Alarms_Events"].query("asset_id == @asset_id").sort_values("timestamp")
    return dict(
        asset=records(master)[0],
        state=records(state)[0],
        telemetry_trend_3d=trend,
        alarms=records(alarms[alarms.severity.isin(["high", "critical", "medium"])], 10) or records(alarms, 10),
        maintenance=records(s["Maintenance_History"].query("asset_id == @asset_id").sort_values("opened_at"), 8),
        similar_cases=records(s["Historical_Service_Cases"].query("asset_class == @cls and historical_issue_code == @issue"))
        or records(s["Historical_Service_Cases"].query("asset_class == @cls"), 3),
        docs=records(s["Knowledge_Docs"].query("doc_id == @doc or scope == @cls")),
        data_quality=records(s["Data_Quality"].query("asset_id == @asset_id")),
    )


SYSTEM = """You are the Valmet field-service copilot inside a 3D digital twin of a boiler room.
Answer ONLY from the CONTEXT (synthetic Excel builder pack). If something is not in it, say so.
Cite evidence IDs in square brackets: [ASSET-…], [EVT-…], [WO-…], [CASE-…], [DOC-…], [DQ-…].
On arrival give a compact report with 4 headed sections:
1. What is wrong — diagnosis, health score, model confidence, abnormal telemetry (first -> last).
2. Why — linked alarms and maintenance history.
3. What to do next — numbered, safe inspection steps (lockout/permit first) from cases and docs.
4. Data trust — data-quality flags the technician must verify; overall confidence.
Keep it under 250 words. Answer in the language of the user's question."""


def claude_client():
    try:
        import anthropic
        client = anthropic.Anthropic()
        return client
    except Exception:
        return None


def ask_claude(client, ctx, history, question):
    messages = history + [{"role": "user", "content": question}]
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "low"},
        system=SYSTEM + "\n\nCONTEXT:\n" + json.dumps(ctx, ensure_ascii=False),
        messages=messages,
    )
    if resp.stop_reason == "refusal":
        return "The model declined this request."
    return "".join(b.text for b in resp.content if b.type == "text")


def offline_report(ctx, question):
    """Fallback without an LLM: the same evidence, templated. Every line carries its source ID."""
    a, st = ctx["asset"], ctx["state"]
    out = [f"**{a['tag']} [{a['asset_id']}] — {a['asset_class']}**, {a['area_name']} ({a['criticality']})", "",
           "**1. What is wrong**",
           f"- {st['issue_summary']} (`{st['issue_code']}`), condition {st['condition_state']}, "
           f"health {st['health_score']}/100, model confidence {st['model_confidence']}, urgency {st['urgency']}."]
    for k, v in ctx["telemetry_trend_3d"].items():
        if v.get("first") is not None and v["first"] != 0:
            ch = (v["last"] - v["first"]) / abs(v["first"]) * 100
            if abs(ch) > 8:
                out.append(f"- `{k}`: {v['first']} -> {v['last']} ({ch:+.0f}%, range {v['min']}–{v['max']})")
    out += ["", "**2. Why (evidence)**"]
    out += [f"- [{e['event_id']}] {e['event_type']} / {e['event_code']} ({e['severity']}) {e['timestamp'][:16]}"
            for e in ctx["alarms"][-5:]] or ["- No alarms recorded."]
    out += [f"- [{w['work_order_id']}] {w['action']} → {w['result']} ({w['opened_at'][:10]})" for w in ctx["maintenance"][-3:]]
    out += ["", "**3. What to do next**", "1. Obtain permit, apply lockout/tagout, confirm asset identity on site."]
    for i, c in enumerate(ctx["similar_cases"][:2], 2):
        out.append(f"{i}. {c['recommended_next_check'].capitalize()} — {c['historical_interpretation']} [{c['case_id']}]")
    for d in ctx["docs"][:1]:
        out.append(f"{len(ctx['similar_cases'][:2]) + 2}. {d['content']} [{d['doc_id']}]")
    out += ["", "**4. Data trust**"]
    out += [f"- [{q['issue_id']}] {q['issue_type']} ({q['severity']}) in {q['source_sheet']} → {q['recommended_handling']}"
            for q in ctx["data_quality"]] or ["- No data-quality flags."]
    if question and not question.startswith("ARRIVED"):
        out += ["", f"_Offline mode: free-form question “{question}” needs an LLM — set ANTHROPIC_API_KEY. Evidence above._"]
    return "\n".join(out)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT / "web"), **kw)

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/assets/"):
            try:
                return self._json(200, asset_context(self.path.rsplit("/", 1)[1]))
            except KeyError:
                return self._json(404, {"error": "unknown asset"})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/copilot":
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            ctx = asset_context(str(req["asset_id"]))
        except (KeyError, ValueError):
            return self._json(400, {"error": "asset_id required"})
        question = str(req.get("question") or "ARRIVED: run full diagnosis.")[:2000]
        history = [m for m in req.get("history", []) if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)][-10:]
        client = claude_client()
        if client:
            try:
                return self._json(200, {"mode": MODEL, "answer": ask_claude(client, ctx, history, question)})
            except Exception as e:  # no credentials / network: degrade to grounded offline report
                print("[copilot] LLM unavailable:", type(e).__name__, e)
        return self._json(200, {"mode": "offline-evidence", "answer": offline_report(ctx, question)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Valmet demo on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

"""Interactive UI for the Bayesian Hallucination Regime Lab.

  python3 app.py            # then open http://localhost:8765
  python3 app.py --port 9000 --results results/clinical_exploratory

Live mode needs Ollama with qwen3:4b-instruct. Replay, Results and How-it-works work offline.
Standard library only (http.server); the page is ui/index.html.
"""
import argparse
import json
import re
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from regime_lab.data.collector import OllamaGenerator
from regime_lab.data.dossiers import DOMAINS, make_dossier
from regime_lab.data.grader import grade_answer
from regime_lab.usecase.guard import ROW_RE, RegimeGuard
from regime_lab.util import derive_seed, load_report, read_json

ROOT = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--results", default="results/v2_samplerfix/clinical_exploratory")
ap.add_argument("--cost-ratio", type=float, default=10.0)
ap.add_argument("--no-browser", action="store_true")
args = ap.parse_args()

THR = 1 / (1 + args.cost_ratio)
GEN = OllamaGenerator()
GUARD = RegimeGuard(ROOT / args.results, GEN, forecast_threshold=THR, alarm_threshold=THR)
SESSIONS = {}
LOCK = threading.Lock()
TRANSCRIPTS = {}
_tp = ROOT / "results/v2_samplerfix/usecase/guard_transcripts.json"      # corrected-sampler guard run (F11)
if not _tp.exists():
    _tp = ROOT / "results/usecase/guard_transcripts.json"
if _tp.exists():
    TRANSCRIPTS = {d["dossier_id"]: d for d in read_json(_tp)}


def chart_payload(cid):
    vis, gold = make_dossier(cid, "clinical", 2026, "v2")
    rows = []
    for r in vis["rows"]:
        m = ROW_RE.match(r)
        if m:
            rows.append({"name": m["name"], "year": int(m["year"]), "field": m["field"], "code": m["code"]})
    patients = {}
    for r in rows:
        patients.setdefault(f"{r['name']}|{r['year']}", {"name": r["name"], "year": r["year"], "fields": []})["fields"].append(
            {"field": r["field"], "code": r["code"]})
    return {"id": cid, "patients": list(patients.values()), "fields": DOMAINS["clinical"]["fields"],
            "questions": [t["question"] for t in vis["turns"]], "n_rows": len(rows)}


def expected_for(cid, entity, question):
    """Gold answer for any question resolved to (entity, field); None if unresolvable."""
    vis, gold = make_dossier(cid, "clinical", 2026, "v2")
    fields = DOMAINS["clinical"]["fields"]
    field = max((f for f in fields if f in question), key=len, default=None)
    if not entity or field is None:
        return None, None
    names = [n for n in vis["entities"]]
    if entity[0] not in names:
        return None, None
    e, f = names.index(entity[0]), fields.index(field)
    hidden = f"{e}:{f}" in gold["hidden"]
    code = gold["all_codes"][f"{e}:{f}"]
    return ({"answer": "UNKNOWN" if hidden else code, "hidden_code": code if hidden else None}, gold["all_codes"])


def results_payload():
    out = {}
    hp = ROOT / "results/headline.md"
    if hp.exists():
        out["headline"] = [re.sub(r"^\d+\.\s*", "", l) for l in hp.read_text().splitlines() if re.match(r"^\d+\.", l)]
    for k in ("registry", "registry_exploratory", "clinical", "clinical_exploratory"):
        p = ROOT / f"results/v2_samplerfix/{k}/report.json"      # corrected sampler (variance-sampler fix)
        if p.exists():
            r = load_report(p.parent)        # [F1-diag v1] full-draw diagnostics when present
            out[k] = {"detection_onset": {m: {"auroc": v["auroc"], "ci": v["auroc_ci"]} for m, v in r["detection_onset"].items()},
                      "paired": r["paired_onset_auroc"], "factor_contrasts": r["factor_contrasts"],
                      "persistence": {kk: r["persistence_H2"][kk] for kk in ("P(H_t|H_t-1)", "P(H_t|notH_t-1)", "HH_observed",
                                                                            "HH_null_mean", "p_value")},
                      "onsets": r["onset_test"]["onsets"], "label_counts": r["label_counts"],
                      "converged": {m: v["converged"] for m, v in r["diagnostics"].items()}}
    for k in ("registry", "clinical"):
        p = ROOT / f"results/v2_samplerfix/{k}/intervention.json"
        if p.exists():
            out[f"intervention_{k}"] = read_json(p)
    for k, p in (("attacks", "results/v2_samplerfix/adversarial/attacks.json"),
                 ("guard", "results/v2_samplerfix/usecase/guard_summary.json")):
        if (ROOT / p).exists():
            j = read_json(ROOT / p)
            out[k] = j.get("conditions", j) if k == "attacks" else j
    return out


def ollama_ok():
    try:
        info = GEN.info()
        return "error" not in info, info
    except Exception as exc:  # pragma: no cover
        return False, {"error": repr(exc)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._send(200, (ROOT / "ui/index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/status":
                ok, info = ollama_ok()
                return self._send(200, {"ollama": ok, "model": info.get("model"), "quant": info.get("quantization"),
                                        "forecast_threshold": THR, "alarm_threshold": THR, "cost_ratio": args.cost_ratio,
                                        "results": args.results, "draws": GUARD.S})
            if path == "/api/charts":
                lst = []
                for cid, d in sorted(TRANSCRIPTS.items()):
                    lst.append({"id": cid, "unguarded_H": sum(r["label"] == "H" for r in d["unguarded"]),
                                "guarded_H": sum(r["label"] == "H" for r in d["guarded"]),
                                "holds": sum(r["hold"] for r in d["guarded"])})
                return self._send(200, {"replay": lst})
            m = re.match(r"^/api/chart/(\d+)$", path)
            if m:
                return self._send(200, chart_payload(int(m[1])))
            m = re.match(r"^/api/replay/(\d+)$", path)
            if m:
                cid = int(m[1])
                if cid not in TRANSCRIPTS:
                    return self._send(404, {"error": "no saved run for this chart"})
                _, gold = make_dossier(cid, "clinical", 2026, "v2")
                runs = {}
                for mode, turns in TRANSCRIPTS[cid].items():
                    if mode == "dossier_id":
                        continue
                    ch = GUARD.replay_chain(turns)            # recomputed hidden-Markov quantities (no LLM)
                    runs[mode] = [{**t, **c} for t, c in zip(turns, ch)]
                return self._send(200, {"chart": chart_payload(cid), "runs": runs,
                                        "expected": [g["answer"] for g in gold["turns"]]})
            if path == "/api/results":
                return self._send(200, results_payload())
            return self._send(404, {"error": "not found"})
        except Exception as exc:
            return self._send(500, {"error": repr(exc)})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            b = self._body()
            if path == "/api/session":
                cid = int(b.get("chart", 20003))
                vis, _ = make_dossier(cid, "clinical", 2026, "v2")
                sid = uuid.uuid4().hex[:12]
                with LOCK:
                    SESSIONS[sid] = {"chart": cid, "planned": [t["question"] for t in vis["turns"]],
                                     "sess": GUARD.session(vis["system"], vis["rows"], "clinical", bool(b.get("guard", True)),
                                                           bool(b.get("reground", False)),
                                                           "v2" if b.get("coref") == "v2" else "v1"),   # [W9 v2]
                                     "lock": threading.Lock(), "pre": None}
                return self._send(200, {"session": sid, "chart": chart_payload(cid)})
            S = SESSIONS.get(b.get("session"))
            if S is None:
                return self._send(400, {"error": "unknown session"})
            if path == "/api/pre":
                q = str(b.get("question", "")).strip()[:300]
                if not q:
                    return self._send(400, {"error": "empty question"})
                with S["lock"]:
                    pre = S["sess"].pre(q)
                    S["pre"] = pre
                return self._send(200, pre)
            if path == "/api/answer":
                with S["lock"]:
                    pre = S["pre"]
                    if pre is None:
                        return self._send(400, {"error": "call /api/pre first"})
                    t = pre["turn"]
                    planned = S["planned"][t - 1] if t <= len(S["planned"]) else None
                    seed = (derive_seed(2026, "clinical", S["chart"], t) if planned == pre["question"]
                            else derive_seed("custom", S["chart"], t, pre["question"]))
                    post = S["sess"].answer(seed)
                    S["pre"] = None
                exp, codes = expected_for(S["chart"], pre["entity"], pre["question"])
                grade = None
                if exp is not None:
                    grade = {**grade_answer(post["answer"], exp, codes), "expected": exp["answer"]}
                return self._send(200, {**post, "grade": grade, "seed": seed})
            return self._send(404, {"error": "not found"})
        except Exception as exc:
            return self._send(500, {"error": repr(exc)})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    url = f"http://localhost:{args.port}"
    print(f"Hallucination Regime Lab UI → {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass

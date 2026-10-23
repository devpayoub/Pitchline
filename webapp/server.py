"""
AIFootballPredictions — local control panel.

    python webapp/server.py            # http://127.0.0.1:8000

Three buttons map to three job pipelines (download / train / predict). Every step
runs as a real subprocess from the project root, and its output is streamed to the
browser through a polled log buffer.
"""
import os
import sys
import json
import time
import threading
import subprocess
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PY = sys.executable
PORT = int(os.environ.get("PORT", 8000))

# --------------------------------------------------------------------------- #
# Job definitions
# --------------------------------------------------------------------------- #
LEAGUES = "E0 I1 SP1 F1 D1"
EXTRA = "N1 B1 P1 T1 G1"
SEASONS = "2627 2526 2425 2324"

JOBS = {
    "download": {
        "title": "Download data",
        "steps": [
            ("Downloading the five main leagues", f'scripts/data_acquisition.py --leagues {LEAGUES} --seasons {SEASONS} --raw_data_output_dir data/raw'),
            ("Downloading five extra leagues for the CL model", f'scripts/data_acquisition.py --leagues {EXTRA} --seasons {SEASONS} --raw_data_output_dir data/raw'),
            ("Pooling ten leagues into the European dataset", "scripts/build_european_dataset.py"),
            ("Preprocessing and feature engineering", "scripts/data_preprocessing.py --raw_data_input_dir data/raw --processed_data_output_dir data/processed --num_features 20 --clustering_threshold 0.5 --target over25"),
            ("Fetching upcoming fixtures (7-day window)", "scripts/acquire_next_matches.py --get_teams_names_dir data/processed --next_matches_output_file data/next_matches.json"),
            ("Fetching Champions League fixtures (16-day window)", "scripts/acquire_next_matches.py --get_teams_names_dir data/processed --next_matches_output_file data/next_matches_16d.json --window_days 16"),
            ("Merging CL fixtures into the main file", "scripts/merge_cl.py"),
            ("Fetching Europa League fixtures", "scripts/fetch_europa_league.py --window_days 7"),
        ],
        "timeout": 1800,
    },
    "train": {
        "title": "Train models",
        "steps": [
            ("Training O/U 2.5 models (5-fold)", "scripts/train_models.py --processed_data_input_dir data/processed --trained_models_output_dir models --metric_choice accuracy --n_splits 5 --voting soft --target over25"),
            ("Training 1X2 models (3-fold)", "scripts/train_models.py --processed_data_input_dir data/processed --trained_models_output_dir models --metric_choice accuracy --n_splits 3 --voting soft --target result"),
            ("Training BTTS / O-U 1.5 / O-U 3.5 models", "scripts/train_additional_targets.py"),
            ("Measuring out-of-time accuracy", "scripts/backtest_models.py"),
        ],
        "timeout": 5400,
    },
    "predict": {
        "title": "Predict & compare",
        "steps": [
            ("Generating the O/U 2.5 + 1X2 card", "scripts/make_predictions.py --input_leagues_models_dir models --input_data_predict_dir data/processed --final_predictions_out_file final_predictions.txt --next_matches data/next_matches.json"),
            ("Generating the full 7-market card", "scripts/make_predictions_full.py --input_leagues_models_dir models --input_data_predict_dir data/processed --final_predictions_out_file data/final_predictions_full.txt --next_matches data/next_matches.json"),
            ("Exporting model probabilities", "scripts/export_results.py model"),
            ("Fetching live odds (1X2, O/U 2.5, BTTS)", "scripts/fetch_odds_v4.py --predictions data/results_model.json"),
            ("Comparing model vs market", "scripts/export_results.py merge"),
            ("Archiving this run for later grading", "scripts/archive_predictions.py"),
            ("Building the shareable report", "scripts/build_report_page.py"),
        ],
        "timeout": 1800,
    },
    "grade": {
        "title": "Grade results",
        "steps": [
            ("Collecting finished matches", "scripts/grade_predictions.py"),
        ],
        "timeout": 900,
    },
}


class Runner:
    """Single worker: one job at a time, append-only log, polled by the UI."""

    def __init__(self):
        self.lock = threading.Lock()
        self.job = None
        self.status = "idle"            # idle | running | done | failed | stopped
        self.logs = []
        self.step = 0
        self.total = 0
        self.started = None
        self.finished = None
        self.error = None
        self.process = None
        self.stop_requested = False

    # -- helpers -----------------------------------------------------------
    def log(self, text, level="out"):
        with self.lock:
            for line in str(text).rstrip("\n").split("\n"):
                self.logs.append({"t": time.time(), "level": level, "text": line})
            if len(self.logs) > 4000:
                self.logs = self.logs[-3000:]

    def state(self):
        with self.lock:
            return {
                "job": self.job,
                "status": self.status,
                "step": self.step,
                "total": self.total,
                "started": self.started,
                "finished": self.finished,
                "error": self.error,
                "log_count": len(self.logs),
                "results_ready": os.path.exists(os.path.join(BASE, "data", "results.json")),
                "report_ready": os.path.exists(os.path.join(BASE, "data", "predictions_report.html")),
            }

    # -- execution ---------------------------------------------------------
    def start(self, name):
        with self.lock:
            if self.status == "running":
                return False, "A job is already running."
            self.job = name
            self.status = "running"
            self.logs = []
            self.step = 0
            self.total = len(JOBS[name]["steps"])
            self.started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.finished = None
            self.error = None
            self.stop_requested = False
        threading.Thread(target=self._run, args=(name,), daemon=True).start()
        return True, "started"

    def stop(self):
        with self.lock:
            if self.status != "running":
                return False
            self.stop_requested = True
            proc = self.process
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        return True

    def _run(self, name):
        job = JOBS[name]
        env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.log(f"$ {job['title']}", "head")
        try:
            for idx, (label, cmd) in enumerate(job["steps"], start=1):
                if self.stop_requested:
                    raise RuntimeError("stopped by user")
                self.step = idx - 1
                self.log(f"[{idx}/{self.total}] {label}", "step")
                full = f'"{PY}" {cmd}'
                self.log(f"    $ {cmd}", "cmd")
                proc = subprocess.Popen(
                    full, shell=True, cwd=BASE, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                )
                self.process = proc
                for line in proc.stdout:
                    self.log(line)
                code = proc.wait()
                self.process = None
                if code != 0:
                    raise RuntimeError(f"step {idx} failed with exit code {code}")
                self.step = idx
            self.status = "done"
            self.log("All steps finished.", "ok")
        except Exception as exc:
            if str(exc) == "stopped by user":
                self.status = "stopped"
                self.log("Stopped.", "warn")
            else:
                self.status = "failed"
                self.error = str(exc)
                self.log(f"FAILED: {exc}", "err")
        finally:
            self.finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


RUNNER = Runner()

# Live scores are scraped on demand and cached briefly: one request per poll, not
# one per browser tab.
LIVE_CACHE = {"at": 0.0, "payload": None}
LIVE_LOCK = threading.Lock()
LIVE_TTL = 45


def live_payload(force=False):
    with LIVE_LOCK:
        fresh = LIVE_CACHE["payload"] is not None and (time.time() - LIVE_CACHE["at"]) < LIVE_TTL
        if fresh and not force:
            return LIVE_CACHE["payload"]
        try:
            subprocess.run([PY, "scripts/live_predictions.py"], cwd=BASE,
                           capture_output=True, text=True, timeout=180)
        except Exception as exc:
            if LIVE_CACHE["payload"] is not None:
                return LIVE_CACHE["payload"]
            return {"error": str(exc)}
        target = os.path.join(BASE, "data", "history", "live.json")
        if not os.path.exists(target):
            return {"error": "no live data yet"}
        with open(target, encoding="utf-8") as f:
            payload = json.load(f)
        LIVE_CACHE.update(at=time.time(), payload=payload)
        return payload


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


class Handler(BaseHTTPRequestHandler):
    server_version = "AIFP/1.0"

    def log_message(self, *args):
        pass

    def _send(self, code, body=b"", ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        path = self.path.split("?")[0]
        query = {}
        if "?" in self.path:
            for part in self.path.split("?", 1)[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    query[k] = v

        if path == "/api/state":
            return self._json(RUNNER.state())

        if path == "/api/logs":
            since = int(query.get("since", 0))
            with RUNNER.lock:
                lines = RUNNER.logs[since:]
                total = len(RUNNER.logs)
            return self._json({"since": since, "total": total, "lines": lines})

        if path == "/api/live":
            return self._json(live_payload(force=query.get("force") == "1"))

        if path == "/api/accuracy":
            payload = {}
            for key, name in (("backtest", "backtest.json"), ("graded", "graded.json")):
                target = os.path.join(BASE, "data", "history", name)
                payload[key] = None
                if os.path.exists(target):
                    with open(target, encoding="utf-8") as f:
                        payload[key] = json.load(f)
            index_path = os.path.join(BASE, "data", "history", "index.json")
            payload["archives"] = []
            if os.path.exists(index_path):
                with open(index_path, encoding="utf-8") as f:
                    payload["archives"] = json.load(f)
            return self._json(payload)

        if path == "/api/results":
            target = os.path.join(BASE, "data", "results.json")
            if not os.path.exists(target):
                return self._json({"error": "no results yet"}, 404)
            with open(target, "rb") as f:
                return self._send(200, f.read())

        if path == "/report":
            target = os.path.join(BASE, "data", "predictions_report.html")
            if not os.path.exists(target):
                return self._send(404, b"report not built yet", "text/plain; charset=utf-8")
            with open(target, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")

        if path in ("/", "/index.html"):
            path = "/index.html"

        safe = os.path.normpath(path.lstrip("/")).replace("\\", "/")
        if safe.startswith(".."):
            return self._send(403, b"forbidden", "text/plain")
        target = os.path.join(STATIC, safe)
        if not os.path.isfile(target):
            return self._send(404, b"not found", "text/plain")
        with open(target, "rb") as f:
            body = f.read()
        return self._send(200, body, MIME.get(os.path.splitext(target)[1], "application/octet-stream"))

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)

        if path.startswith("/api/run/"):
            name = path.rsplit("/", 1)[-1]
            if name not in JOBS:
                return self._json({"error": "unknown job"}, 400)
            ok, msg = RUNNER.start(name)
            return self._json({"ok": ok, "message": msg}, 200 if ok else 409)

        if path == "/api/stop":
            return self._json({"ok": RUNNER.stop()})

        return self._json({"error": "not found"}, 404)


def main():
    print("=" * 62)
    print("  AI Football Predictions — control panel")
    print(f"  http://127.0.0.1:{PORT}")
    print("=" * 62)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

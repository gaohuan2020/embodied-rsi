from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

RUN_ID = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dumps(value) + "\n", encoding="utf-8")
    temporary.replace(path)


class Store:
    def __init__(self, root="artifacts"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "monitor.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                  id TEXT PRIMARY KEY, kind TEXT, backend TEXT, status TEXT,
                  created REAL, updated REAL, config TEXT, summary TEXT);
                CREATE TABLE IF NOT EXISTS events (
                  seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL,
                  kind TEXT, payload TEXT);
                CREATE INDEX IF NOT EXISTS event_run ON events(run_id, seq);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def create(self, kind, backend, config, run_id=None):
        run_id = run_id or f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid run id")
        now = time.time()
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                       (run_id, kind, backend, "running", now, now, dumps(config), "{}"))
        self.run_dir(run_id).mkdir(parents=True)
        self.event(run_id, "started", {"backend": backend})
        return run_id

    def run_dir(self, run_id):
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid run id")
        return self.root / "runs" / run_id

    def event(self, run_id, kind, payload):
        now = time.time()
        with self.connect() as db:
            db.execute("INSERT INTO events(run_id,ts,kind,payload) VALUES (?,?,?,?)",
                       (run_id, now, kind, dumps(payload)))
            db.execute("UPDATE runs SET updated=? WHERE id=?", (now, run_id))

    def finish(self, run_id, status="completed", summary=None):
        self.event(run_id, status, summary or {})
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?,summary=?,updated=? WHERE id=?",
                       (status, dumps(summary or {}), time.time(), run_id))

    def runs(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY created DESC LIMIT 200").fetchall()
        result = []
        for row in rows:
            r = dict(row)
            r["config"], r["summary"] = json.loads(r["config"]), json.loads(r["summary"])
            r["stale"] = r["status"] == "running" and time.time() - r["updated"] > 60
            result.append(r)
        return result

    def events(self, run_id, after=0, limit=5000):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                              (run_id, after, limit)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

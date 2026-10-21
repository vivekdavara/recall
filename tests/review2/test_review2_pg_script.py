"""Review 2 findings for scripts/pg.sh (DESIGN.md "The local database"). Each test fails on commit 0e87c41.

Runs a throwaway cluster under the checkout's .scratch/ on a free port, as tests/test_pg_script.py does."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "pg.sh"
PG_BIN = os.environ.get("PG_BIN", "/opt/homebrew/opt/postgresql@17/bin")

if not (Path(PG_BIN) / "initdb").exists():
    pytest.skip(f"no PostgreSQL 17 at {PG_BIN}", allow_module_level=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_a_relative_data_directory_starts():
    """DESIGN.md: "RECALL_PG_DATA ... override[s] ... the data directory". A relative one (RECALL_PG_DATA=rel/data)
    initialises fine but the server can't start: unix_socket_directories='rel/data' is resolved by the postmaster
    relative to its data directory, so the log says 'could not create lock file "rel/data/.s.PGSQL.PORT.lock": No
    such file or directory'. The script should make the path absolute (or refuse a relative one with a message)."""
    work = ROOT / ".scratch" / f"r2-{uuid.uuid4().hex[:6]}"
    work.mkdir(parents=True)
    env = {**os.environ, "PG_BIN": PG_BIN, "RECALL_PG_PORT": str(free_port()), "RECALL_PG_DATA": "rel/data"}
    try:
        r = subprocess.run([str(SCRIPT), "start"], cwd=work, env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
    finally:
        subprocess.run([str(SCRIPT), "stop"], cwd=work, env=env, capture_output=True, timeout=60)
        shutil.rmtree(work, ignore_errors=True)

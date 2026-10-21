"""scripts/pg.sh end to end, on a throwaway cluster (its own data directory and a free port).

It needs PostgreSQL 17 and pgvector at $PG_BIN (Homebrew's by default). CI's pg-script job installs them and sets
RECALL_REQUIRE_PG_SCRIPT=1, so there it fails rather than skips when they are missing."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "pg.sh"
PG_BIN = os.environ.get("PG_BIN", "/opt/homebrew/opt/postgresql@17/bin")

if not (Path(PG_BIN) / "initdb").exists() and os.environ.get("RECALL_REQUIRE_PG_SCRIPT") != "1":
    pytest.skip(f"no PostgreSQL 17 at {PG_BIN} (CI's pg-script job runs these)", allow_module_level=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def cluster() -> Iterator[tuple[dict[str, str], Path, int]]:
    # Under the repository, not the system temp directory: the socket lives in the data directory, and a Unix
    # socket path must be short (about 100 bytes on macOS); pytest's tmp_path is too long for it.
    # A parent directory that doesn't exist yet, as in a fresh checkout (CI caught start failing on one). Short:
    # the server's socket is <data>/.s.PGSQL.<port>, and a Unix socket path may be at most 103 bytes on macOS.
    data = ROOT / ".scratch" / f"pg{uuid.uuid4().hex[:4]}" / "d"
    socket_path = f"{data}/.s.PGSQL.{65535}"
    if len(socket_path.encode()) > 103:
        pytest.fail(f"the checkout is too deep for a Unix socket path here ({len(socket_path.encode())} bytes)")
    port = free_port()
    env = {**os.environ, "PG_BIN": PG_BIN, "RECALL_PG_PORT": str(port), "RECALL_PG_DATA": str(data)}
    try:
        yield env, data, port
    finally:
        subprocess.run([str(SCRIPT), "stop"], env=env, capture_output=True, timeout=60)
        shutil.rmtree(data.parent, ignore_errors=True)


def run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=120)


def test_start_status_psql_stop_and_reset(cluster: tuple[dict[str, str], Path, int]):
    env, data, port = cluster
    r = run(env, "start")
    assert (r.returncode, r.stdout) == (0, f"started on port {port}\n"), r.stderr
    assert run(env, "start").stdout == f"already running on port {port}\n"
    assert run(env, "status").stdout == f"running on port {port}\n"
    assert run(env, "url").stdout == f"postgresql://localhost:{port}/recall\n"

    def sql(query: str) -> str:
        r = run(env, "psql", "-Atc", query)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()

    assert sql("select string_agg(datname, ',' order by datname) from pg_database where datname like 'recall%'") == (
        "recall,recall_test"
    )
    assert (
        sql("select datlocprovider::text || ' ' || datlocale from pg_database where datname = 'recall'") == "b C.UTF-8"
    )
    assert sql("show listen_addresses") == "localhost"
    assert sql("select 'b' < 'B', 'a b' < 'ab'") == "f|t"  # code point order
    assert sql("create extension vector; select '[1,2]'::vector <-> '[1,2]'::vector") == "CREATE EXTENSION\n0"
    assert (data / "PG_VERSION").exists() and (data / "initdb.log").exists()

    assert run(env, "stop").stdout == "stopped\n"
    r = run(env, "status")
    assert (r.returncode, r.stdout) == (3, "not running\n")
    assert run(env, "stop").stdout == "not running\n"

    # A second start reuses the data directory: the extension is still there.
    assert run(env, "start").stdout == f"started on port {port}\n"
    assert sql("select extname from pg_extension where extname = 'vector'") == "vector"

    r = run(env, "reset")
    assert (r.returncode, r.stdout) == (0, f"removed {data}\n")
    assert not data.exists()
    assert run(env, "reset").stdout == f"nothing at {data}\n"


def test_reset_leaves_a_directory_that_isnt_a_cluster_alone(cluster: tuple[dict[str, str], Path, int]):
    env, data, _ = cluster
    data.mkdir(parents=True)
    (data / "precious.txt").write_text("keep me")
    r = run(env, "reset")
    assert r.returncode == 1 and "left alone" in r.stderr
    assert (data / "precious.txt").read_text() == "keep me"


def test_an_unknown_command_prints_the_usage(cluster: tuple[dict[str, str], Path, int]):
    env, _, _ = cluster
    r = run(env, "launch")
    assert r.returncode == 2 and r.stderr.startswith("usage:")

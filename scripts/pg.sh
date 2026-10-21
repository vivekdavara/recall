#!/usr/bin/env bash
# Recall's project-local PostgreSQL 17 cluster: port 5544, data and socket in .pgdata/ (gitignored).
# See DESIGN.md "The local database". Trust authentication on localhost only: never expose this cluster.
#
#   scripts/pg.sh start    initdb on first use, start, create the recall and recall_test databases
#   scripts/pg.sh stop | status | url | psql [args] | reset
#
# PG_BIN (the PostgreSQL 17 binaries), RECALL_PG_PORT and RECALL_PG_DATA override the defaults; the tests use them
# to run a throwaway cluster.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
bin=${PG_BIN:-/opt/homebrew/opt/postgresql@17/bin}
port=${RECALL_PG_PORT:-5544}
data=${RECALL_PG_DATA:-$root/.pgdata}

# Without a valid LC_ALL, macOS's postmaster can become multithreaded during startup and refuse to start.
export LC_ALL=${LC_ALL:-en_US.UTF-8}

pg() { "$bin/$1" "${@:2}"; }

running() { [ -f "$data/postmaster.pid" ] && pg pg_ctl -D "$data" status >/dev/null 2>&1; }

start() {
    if [ ! -f "$data/PG_VERSION" ]; then
        mkdir -p "$(dirname "$data")"  # initdb's log goes beside the data directory until it exists
        # The builtin C.UTF-8 locale sorts by code point on every platform, so text order here matches CI's
        # (the service container is initialised with the same arguments).
        pg initdb -D "$data" --auth=trust --encoding=UTF8 --locale-provider=builtin --builtin-locale=C.UTF-8 \
            >"$data.initdb.log" 2>&1 || { cat "$data.initdb.log" >&2; exit 1; }
        mv "$data.initdb.log" "$data/initdb.log"
    fi
    if running; then
        echo "already running on port $port"
    else
        pg pg_ctl -D "$data" -l "$data/log" -w \
            -o "-p $port -c listen_addresses=localhost -c unix_socket_directories='$data'" start >/dev/null
        echo "started on port $port"
    fi
    for db in recall recall_test; do
        if [ -z "$(pg psql -h localhost -p "$port" -d postgres -Atc "select 1 from pg_database where datname = '$db'")" ]; then
            pg createdb -h localhost -p "$port" "$db"
        fi
    done
    if [ -z "$(pg psql -h localhost -p "$port" -d recall -Atc "select 1 from pg_available_extensions where name = 'vector'")" ]; then
        echo "the vector extension is not installed for $bin (brew install pgvector)" >&2
        exit 1
    fi
}

case "${1:-}" in
    start) start ;;
    stop)
        if running; then pg pg_ctl -D "$data" -m fast -w stop >/dev/null; echo "stopped"; else echo "not running"; fi ;;
    status)
        if running; then echo "running on port $port"; else echo "not running"; exit 3; fi ;;
    url) echo "postgresql://localhost:$port/recall" ;;
    psql) shift; exec "$bin/psql" -h localhost -p "$port" -d recall "$@" ;;
    reset)
        # The data directory can come from the environment: delete only what is a cluster.
        if [ ! -e "$data" ]; then echo "nothing at $data"; exit 0; fi
        if [ ! -f "$data/PG_VERSION" ]; then echo "not a PostgreSQL data directory, left alone: $data" >&2; exit 1; fi
        if running; then pg pg_ctl -D "$data" -m fast -w stop >/dev/null; fi
        rm -rf "$data"
        echo "removed $data" ;;
    *) echo "usage: $0 start|stop|status|url|psql|reset" >&2; exit 2 ;;
esac

#!/bin/bash
# ============================================================
# HALO Xeon NVMe Recommended Query Runner
# Target: Intel Xeon Silver 4310 + NVMe Storage
# Generated: 2026-02-26
# ============================================================
#
# Usage:
#   chmod +x run_xeon_nvme.sh
#   ./run_xeon_nvme.sh 2>&1 | tee xeon_nvme_results.log
#
# Prerequisites:
#   - MySQL 8.0 running with correct data directory
#   - TPC-H SF30 and JOB (IMDB) databases loaded
#   - Sufficient disk space for results
#
# ============================================================

MYSQL_CMD="mysql -u root"
DB_TPCH="tpch_sf30"
DB_JOB="imdbload"
TIMEOUT=900  # 15 minutes per query
RESULT_DIR="/root/halo_xeon_results/NVMe"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$RESULT_DIR/TPCH" "$RESULT_DIR/JOB"

echo "============================================================"
echo "HALO Xeon NVMe Benchmark Runner"
echo "Started: $(date)"
echo "Timeout: ${TIMEOUT}s per query"
echo "============================================================"

run_query() {
    local db="$1"
    local sql_file="$2"
    local benchmark="$3"
    local query_name=$(basename "$sql_file" .sql)
    local out_file="$RESULT_DIR/$benchmark/${query_name}_result.txt"

    echo ""
    echo "--- [$benchmark] $query_name ---"

    # Cold start: restart MySQL and clear OS caches
    systemctl restart mysql 2>/dev/null || service mysql restart 2>/dev/null
    sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
    sleep 3

    # Run with EXPLAIN ANALYZE
    local start_time=$(date +%s%N)
    timeout $TIMEOUT $MYSQL_CMD "$db" -e "$(cat "$sql_file")" > "$out_file" 2>&1
    local exit_code=$?
    local end_time=$(date +%s%N)
    local elapsed=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc)

    if [ $exit_code -eq 124 ]; then
        echo "  ⏰ TIMEOUT (${TIMEOUT}s) | $query_name"
        echo "TIMEOUT" >> "$out_file"
    elif [ $exit_code -ne 0 ]; then
        echo "  ❌ ERROR (exit=$exit_code) | $query_name"
    else
        echo "  ✅ ${elapsed}s | $query_name"
    fi

    echo "elapsed=${elapsed}s exit_code=${exit_code}" >> "$out_file"
}

# ── Run TPCH Queries ──
echo ""
echo "=== TPCH Benchmark (SF30) ==="
for sql_file in "$SCRIPT_DIR/NVMe/TPCH/"*.sql; do
    [ -f "$sql_file" ] && run_query "$DB_TPCH" "$sql_file" "TPCH"
done

# ── Run JOB Queries ──
echo ""
echo "=== JOB Benchmark (IMDB) ==="
for sql_file in "$SCRIPT_DIR/NVMe/JOB/"*.sql; do
    [ -f "$sql_file" ] && run_query "$DB_JOB" "$sql_file" "JOB"
done

echo ""
echo "============================================================"
echo "Completed: $(date)"
echo "Results saved to: $RESULT_DIR"
echo "============================================================"

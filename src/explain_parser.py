# -*- coding: utf-8 -*-
"""
Halo Phase 1: MySQL EXPLAIN ANALYZE Parser + Data ETL

Parses MySQL 8.0 EXPLAIN ANALYZE output from collected experiment logs.
Extracts operator instances with all features needed for OLHS sigma learning.

Key data formats handled:
  1. Single-line with literal \\n (Server B JOB/TPCH, Server A NVMe JOB)
  2. Multi-line with real newlines (Server A SATA TPCH)
  3. Both formats produce indented tree with -> prefix per operator

Author: Halo Research Project
"""

import re
import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ── Regex patterns for MySQL EXPLAIN ANALYZE ─────────────────────────

# Matches: -> Operator description  (cost=X rows=Y) (actual time=A..B rows=C loops=D)
# Also handles cost range format: (cost=X..Y rows=Z) used by Materialize/CTE
# Uses \d+\.?\d* instead of [\d.]+ to prevent the '..' separator from being consumed
OPERATOR_PATTERN = re.compile(
    r'^(?P<indent>\s*)'
    r'-> (?P<description>.+?)'
    r'(?:\s+\(cost=(?P<est_cost>\d+\.?\d*)(?:\.\.\d+\.?\d*)?\s+rows=(?P<est_rows>\d+)\))?'
    r'\s*\(actual time=(?P<time_start>\d+\.?\d*)\.\.(?P<time_end>\d+\.?\d*)\s+rows=(?P<actual_rows>\d+)\s+loops=(?P<loops>\d+)\)'
    r'\s*$'
)

# Some operators only have actual time (no cost), e.g. Aggregate, Sort
OPERATOR_PATTERN_NO_COST = re.compile(
    r'^(?P<indent>\s*)'
    r'-> (?P<description>.+?)'
    r'\s+\(actual time=(?P<time_start>\d+\.?\d*)\.\.(?P<time_end>\d+\.?\d*)\s+rows=(?P<actual_rows>\d+)\s+loops=(?P<loops>\d+)\)'
    r'\s*$'
)

# Operator type classification from description
OP_TYPE_PATTERNS = [
    (re.compile(r'^Nested loop inner join', re.I), 'NL_INNER_JOIN'),
    (re.compile(r'^Nested loop left join', re.I), 'NL_LEFT_JOIN'),
    (re.compile(r'^Nested loop semijoin', re.I), 'NL_SEMIJOIN'),
    (re.compile(r'^Nested loop antijoin', re.I), 'NL_ANTIJOIN'),
    (re.compile(r'^Inner hash join', re.I), 'HASH_JOIN'),
    (re.compile(r'^Table scan on <temporary>', re.I), 'TEMP_TABLE_SCAN'),
    (re.compile(r'^Table scan on', re.I), 'TABLE_SCAN'),
    (re.compile(r'^Index scan on', re.I), 'INDEX_SCAN'),
    (re.compile(r'^Index lookup on', re.I), 'INDEX_LOOKUP'),
    (re.compile(r'^Single-row index lookup', re.I), 'SINGLE_ROW_INDEX'),
    (re.compile(r'^Index range scan', re.I), 'INDEX_RANGE_SCAN'),
    (re.compile(r'^Filter:', re.I), 'FILTER'),
    (re.compile(r'^Sort:', re.I), 'SORT'),
    (re.compile(r'^Sort\b', re.I), 'SORT'),
    (re.compile(r'^Aggregate', re.I), 'AGGREGATE'),
    (re.compile(r'^Group aggregate', re.I), 'GROUP_AGGREGATE'),
    (re.compile(r'^Limit:', re.I), 'LIMIT'),
    (re.compile(r'^Stream results', re.I), 'STREAM'),
    (re.compile(r'^Materialize', re.I), 'MATERIALIZE'),
    (re.compile(r'^Hash$', re.I), 'HASH_BUILD'),
    (re.compile(r'^Select #\d+', re.I), 'SUBQUERY'),
]

# Table alias extraction from description
TABLE_ALIAS_PATTERN = re.compile(r'on\s+(\w+)\b')


@dataclass
class OperatorInstance:
    """A single operator instance from an EXPLAIN ANALYZE tree."""
    # Identity
    query_id: str = ''
    hint_id: str = ''
    env_id: str = ''
    tree_position: int = 0  # DFS order

    # Operator info
    op_type: str = ''
    description: str = ''
    table_alias: str = ''

    # Optimizer estimates
    est_cost: float = 0.0
    est_rows: int = 0

    # Actual execution
    actual_time_start: float = 0.0
    actual_time_end: float = 0.0
    actual_rows: int = 0
    loops: int = 1

    # Derived features (computed after tree is built)
    self_time: float = 0.0
    total_work: float = 0.0
    row_est_error: float = 0.0
    cost_per_row: float = 0.0

    # Tree structure
    depth: int = 0
    parent_op_type: str = ''
    parent_idx: int = -1
    num_children: int = 0
    children_indices: List[int] = field(default_factory=list)


def classify_operator(description: str) -> str:
    """Classify operator type from EXPLAIN ANALYZE description."""
    for pattern, op_type in OP_TYPE_PATTERNS:
        if pattern.search(description):
            return op_type
    return 'OTHER'


def extract_table_alias(description: str) -> str:
    """Extract table alias from operator description."""
    match = TABLE_ALIAS_PATTERN.search(description)
    if match:
        alias = match.group(1)
        # Filter out non-table keywords
        if alias.lower() not in {'no', 'using', 'primary', 'condition'}:
            return alias
    return ''


def normalize_explain_text(raw_text: str) -> List[str]:
    """
    Normalize EXPLAIN ANALYZE text to multi-line format.

    Handles two formats:
      1. Single line with literal \\n (most files)
      2. Already multi-line (Server A SATA TPCH)
    """
    # Check if it contains literal \n sequences (escaped newlines)
    if '\\n' in raw_text:
        lines = raw_text.split('\\n')
    else:
        lines = raw_text.split('\n')

    # Clean up each line
    cleaned = []
    for line in lines:
        line = line.strip()
        if line.startswith('EXPLAIN'):
            continue
        if line.startswith('->') or line.startswith('- >'):
            # Some lines have extra spaces
            cleaned.append(line)
        elif line.startswith('Hash') and not line.startswith('Hash '):
            # Hash build marker (child of hash join)
            cleaned.append('    ' * 10 + '-> ' + line)
        elif cleaned and not line.startswith('-'):
            # Continuation of previous line
            if cleaned:
                cleaned[-1] += ' ' + line
    return cleaned


def parse_tree(lines: List[str]) -> List[OperatorInstance]:
    """
    Parse normalized EXPLAIN ANALYZE lines into operator instances.
    Uses indentation to determine tree structure.
    """
    operators = []
    indent_stack = []  # Stack of (indent_level, operator_index)

    for line in lines:
        # Try to match operator pattern (with cost)
        m = OPERATOR_PATTERN.match(line)
        if not m:
            m = OPERATOR_PATTERN_NO_COST.match(line)

        if not m:
            continue

        groups = m.groupdict()
        indent = len(groups.get('indent', ''))
        description = groups['description'].strip()

        op = OperatorInstance(
            tree_position=len(operators),
            op_type=classify_operator(description),
            description=description[:200],  # Truncate long descriptions
            table_alias=extract_table_alias(description),
            est_cost=float(groups.get('est_cost', 0) or 0),
            est_rows=int(groups.get('est_rows', 0) or 0),
            actual_time_start=float(groups['time_start']),
            actual_time_end=float(groups['time_end']),
            actual_rows=int(groups['actual_rows']),
            loops=int(groups['loops']),
            depth=0,
        )

        # Determine parent from indentation
        while indent_stack and indent_stack[-1][0] >= indent:
            indent_stack.pop()

        if indent_stack:
            parent_idx = indent_stack[-1][1]
            op.parent_idx = parent_idx
            op.parent_op_type = operators[parent_idx].op_type
            op.depth = operators[parent_idx].depth + 1
            operators[parent_idx].children_indices.append(len(operators))
            operators[parent_idx].num_children += 1

        indent_stack.append((indent, len(operators)))
        operators.append(op)

    return operators


def compute_self_times(operators: List[OperatorInstance]) -> None:
    """
    Compute self_time for each operator by subtracting children's time.

    self_time = (actual_time_end - actual_time_start) - Σ(children time)

    Note: MySQL actual_time is iterator-based. loops > 1 may cause
    accumulated error. Mitigated by repeated run averaging (documented
    as known limitation in paper).
    """
    for op in operators:
        total_time = op.actual_time_end - op.actual_time_start
        children_time = sum(
            operators[ci].actual_time_end - operators[ci].actual_time_start
            for ci in op.children_indices
        )
        op.self_time = max(0.0, total_time - children_time)
        op.total_work = op.self_time * op.loops

        # Q-error (log ratio of actual vs estimated rows)
        if op.est_rows > 0 and op.actual_rows > 0:
            import math
            op.row_est_error = math.log(op.actual_rows / op.est_rows)
        elif op.est_rows > 0:
            op.row_est_error = -10.0  # Very pessimistic estimate
        else:
            op.row_est_error = 0.0

        # Cost per row
        op.cost_per_row = op.est_cost / max(op.est_rows, 1)


# ── File-level parsing ────────────────────────────────────────────────

@dataclass
class QueryResult:
    """A single query execution result (baseline or hinted)."""
    query_id: str
    hint_variant: str  # e.g. "join_method_1", "optimizer_2", or "baseline"
    status: str
    execution_time_s: float
    operators: List[OperatorInstance]
    raw_explain: str = ''


def parse_explain_file(filepath: str, benchmark: str = 'JOB') -> List[QueryResult]:
    """
    Parse a complete EXPLAIN ANALYZE log file.

    File format:
    =========================================
    Query: 1a_join_method_1 [1/678]    (or "Query: 1a.sql [1/113]" for baseline)
    Timestamp: ...
    =========================================

    [EXPLAIN ANALYZE]
    ----------------------------------------
    EXPLAIN
    <tree>
    ----------------------------------------
    Status: SUCCESS
    Execution Time: X.XXXs
    """
    results = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Split by query blocks
    blocks = re.split(r'={40,}', content)

    i = 0
    while i < len(blocks):
        block = blocks[i].strip()

        # Look for "Query:" header
        query_match = re.search(
            r'Query:\s+(.+?)(?:\s+\[\d+/\d+\])?$',
            block, re.MULTILINE
        )

        if query_match:
            raw_query_name = query_match.group(1).strip()

            # Parse query_id and hint_variant
            query_id, hint_variant = _parse_query_name(raw_query_name, benchmark)

            # Next block should contain EXPLAIN ANALYZE
            i += 1
            if i >= len(blocks):
                break

            explain_block = blocks[i].strip()

            # Extract EXPLAIN tree
            explain_match = re.search(
                r'\[EXPLAIN ANALYZE\]\s*-{30,}\s*(.*?)\s*-{30,}',
                explain_block, re.DOTALL
            )

            if not explain_match:
                # Try without bracketed header
                explain_text = explain_block
            else:
                explain_text = explain_match.group(1).strip()

            # Remove mysql warning
            explain_text = re.sub(
                r'mysql:\s*\[Warning\].*?$', '',
                explain_text, flags=re.MULTILINE
            ).strip()

            # Extract status and execution time
            status = 'UNKNOWN'
            exec_time = 0.0
            status_match = re.search(r'Status:\s+(\w+)', explain_block)
            if status_match:
                status = status_match.group(1)
            time_match = re.search(
                r'Execution Time:\s+([\d.]+)s',
                explain_block
            )
            if time_match:
                exec_time = float(time_match.group(1))

            # Parse the tree
            if explain_text and 'actual time=' in explain_text:
                lines = normalize_explain_text(explain_text)
                operators = parse_tree(lines)
                compute_self_times(operators)

                # Tag operators with identity
                for op in operators:
                    op.query_id = query_id
                    op.hint_id = hint_variant

                results.append(QueryResult(
                    query_id=query_id,
                    hint_variant=hint_variant,
                    status=status,
                    execution_time_s=exec_time,
                    operators=operators,
                    raw_explain=explain_text[:500],
                ))

        i += 1

    return results


def _parse_query_name(raw_name: str, benchmark: str = 'JOB') -> Tuple[str, str]:
    """
    Parse query name into (query_id, hint_variant).

    Examples (JOB):
      "1a.sql"                -> ("1a", "baseline")
      "1a_join_method_1"      -> ("1a", "join_method_1")
    Examples (TPCH):
      "1"                     -> ("tpch_q1", "baseline")
      "1_hint1_3"             -> ("tpch_q1", "hint1_3")
    """
    # Remove .sql suffix
    name = re.sub(r'\.sql$', '', raw_name)

    if benchmark == 'TPCH':
        # TPC-H hinted variant (e.g., "1_hint1_3", "10_hint2_1")
        tpch_hint_match = re.match(r'^(\d+)_(.+)$', name)
        if tpch_hint_match:
            return f"tpch_q{tpch_hint_match.group(1)}", tpch_hint_match.group(2)

        # TPC-H baseline (e.g., "1" or "10")
        tpch_match = re.match(r'^(\d+)$', name)
        if tpch_match:
            return f"tpch_q{tpch_match.group(1)}", "baseline"

        return name, "unknown"
    else:
        # JOB hinted variant (e.g., "1a_join_method_1")
        job_match = re.match(r'^(\d+[a-z]?)_(.+)$', name)
        if job_match:
            return job_match.group(1), job_match.group(2)

        # JOB baseline (e.g., "1a")
        job_baseline = re.match(r'^(\d+[a-z]?)$', name)
        if job_baseline:
            return job_baseline.group(1), "baseline"

        return name, "unknown"


# ── Data inventory and ETL ────────────────────────────────────────────

@dataclass
class DataSource:
    """A data source from our experimental setup."""
    server: str        # "A" or "B"
    storage: str       # "NVMe" or "SATA"
    benchmark: str     # "JOB" or "TPCH"
    hint_id: str       # "baseline", "hint01", "hint02", "hint04", "hint05"
    filepath: str      # Full path to explain_analyze file
    env_id: str = ''   # Computed: "A_NVMe", "A_SATA", etc.

    def __post_init__(self):
        self.env_id = f"{self.server}_{self.storage}"


def discover_data_sources(base_dir: str = '/root/kcj_sqlhint') -> List[DataSource]:
    """
    Auto-discover all EXPLAIN ANALYZE files in the data directory.
    Returns a structured inventory of all data sources.

    Server A has BOTH JOB and TPCH on BOTH NVMe and SATA:
      NVMe: baseline_explain_analyze.txt = JOB
            baseline_cold/ + hint*_cold/ = TPCH
      SATA: original_baseline_ssd/ = JOB
            baseline_cold/ + hint*_cold/ = TPCH
    """
    sources = []

    # ── Server A NVMe ──
    nvme_base = f"{base_dir}/server_a/SQL_hint/NVMe/Original_Result_Logs/NVMe_result/results"
    # JOB baseline (hot start, original)
    _add_if_exists(sources, 'A', 'NVMe', 'JOB', 'baseline',
                   f"{nvme_base}/baseline_explain_analyze.txt")
    # TPCH baseline (cold start)
    _add_if_exists(sources, 'A', 'NVMe', 'TPCH', 'baseline',
                   f"{nvme_base}/baseline_cold/baseline_cold_explain_analyze.txt")
    # TPCH hints (cold start)
    for hint_id, dirname in [
        ('hint01', 'hint01_cold'), ('hint02', 'hint02_cold'),
        ('hint04', 'hint04_cold'), ('hint05', 'hint05_cold')
    ]:
        _add_if_exists(sources, 'A', 'NVMe', 'TPCH', hint_id,
                       f"{nvme_base}/{dirname}/{dirname}_explain_analyze.txt")

    # ── Server A SATA ──
    sata_base = f"{base_dir}/server_a/SQL_hint/SATA/Original_Result_Logs/SSD_result/results"
    # JOB baseline (original)
    _add_if_exists(sources, 'A', 'SATA', 'JOB', 'baseline',
                   f"{sata_base}/original_baseline_ssd/baseline_explain_analyze.txt")
    # TPCH baseline (cold start)
    _add_if_exists(sources, 'A', 'SATA', 'TPCH', 'baseline',
                   f"{sata_base}/baseline_cold/baseline_cold_explain_analyze.txt")
    # TPCH hints (cold start)
    for hint_id, dirname in [
        ('hint01', 'hint01_cold'), ('hint02', 'hint02_cold'),
        ('hint04', 'hint04_cold'), ('hint05', 'hint05_cold')
    ]:
        _add_if_exists(sources, 'A', 'SATA', 'TPCH', hint_id,
                       f"{sata_base}/{dirname}/{dirname}_explain_analyze.txt")

    # ── Server B NVMe (JOB + TPCH) ──
    for benchmark, bdir in [('JOB', 'JOB'), ('TPCH', 'TPCH')]:
        b_nvme = f"{base_dir}/server_b/SQL_hint/Original_Result_Logs/NVMe/{bdir}"
        _add_if_exists(sources, 'B', 'NVMe', benchmark, 'baseline',
                       f"{b_nvme}/baseline_explain_analyze.txt")
        for hint_id, hdir in [
            ('hint01', 'Hint_01_single_hints'),
            ('hint02', 'Hint_02_table_specific'),
            ('hint04', 'Hint_04_join_order_index'),
            ('hint05', 'Hint_05_index_strategy'),
        ]:
            _add_if_exists(sources, 'B', 'NVMe', benchmark, hint_id,
                           f"{b_nvme}/{hdir}/hint_results_explain_analyze.txt")

    # ── Server B SATA (JOB + TPCH) ──
    for benchmark, bdir in [('JOB', 'JOB'), ('TPCH', 'TPCH')]:
        b_sata = f"{base_dir}/server_b/SQL_hint/Original_Result_Logs/SATA/{bdir}"
        _add_if_exists(sources, 'B', 'SATA', benchmark, 'baseline',
                       f"{b_sata}/baseline_explain_analyze.txt")
        for hint_id, hdir in [
            ('hint01', 'Hint_01_single_hints'),
            ('hint02', 'Hint_02_table_specific'),
            ('hint04', 'Hint_04_join_order_index'),
            ('hint05', 'Hint_05_index_strategy'),
        ]:
            _add_if_exists(sources, 'B', 'SATA', benchmark, hint_id,
                           f"{b_sata}/{hdir}/hint_results_explain_analyze.txt")

    return sources


def _add_if_exists(sources, server, storage, benchmark, hint_id, filepath):
    if os.path.exists(filepath):
        sources.append(DataSource(
            server=server, storage=storage, benchmark=benchmark,
            hint_id=hint_id, filepath=filepath
        ))
    else:
        logger.debug(f"Not found: {filepath}")


def run_etl(base_dir: str = '/root/kcj_sqlhint',
            output_dir: str = '/root/halo/data') -> Dict:
    """
    Run the full ETL pipeline:
    1. Discover all data sources
    2. Parse each EXPLAIN ANALYZE file
    3. Extract operator instances
    4. Save as unified parquet + JSON summary

    Returns summary statistics.
    """
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    sources = discover_data_sources(base_dir)

    logger.info(f"Discovered {len(sources)} data sources")

    all_operators = []
    all_queries = []
    errors = []

    for src in sources:
        logger.info(f"Parsing: {src.env_id} / {src.benchmark} / {src.hint_id} "
                     f"({src.filepath})")
        try:
            results = parse_explain_file(src.filepath, benchmark=src.benchmark)
            logger.info(f"  → {len(results)} query results parsed")

            for qr in results:
                # Build query-level record
                all_queries.append({
                    'server': src.server,
                    'storage': src.storage,
                    'env_id': src.env_id,
                    'benchmark': src.benchmark,
                    'hint_set': src.hint_id,
                    'query_id': qr.query_id,
                    'hint_variant': qr.hint_variant,
                    'status': qr.status,
                    'execution_time_s': qr.execution_time_s,
                    'num_operators': len(qr.operators),
                })

                # Build operator-level records
                for op in qr.operators:
                    op.env_id = src.env_id
                    all_operators.append({
                        'server': src.server,
                        'storage': src.storage,
                        'env_id': src.env_id,
                        'benchmark': src.benchmark,
                        'hint_set': src.hint_id,
                        'query_id': op.query_id,
                        'hint_variant': op.hint_id,
                        'tree_position': op.tree_position,
                        'op_type': op.op_type,
                        'table_alias': op.table_alias,
                        'est_cost': op.est_cost,
                        'est_rows': op.est_rows,
                        'actual_time_start': op.actual_time_start,
                        'actual_time_end': op.actual_time_end,
                        'actual_rows': op.actual_rows,
                        'loops': op.loops,
                        'self_time': op.self_time,
                        'total_work': op.total_work,
                        'row_est_error': op.row_est_error,
                        'cost_per_row': op.cost_per_row,
                        'depth': op.depth,
                        'parent_op_type': op.parent_op_type,
                        'num_children': op.num_children,
                        'description': op.description,
                    })

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            errors.append({'source': src.filepath, 'error': str(e)})

    # ── Save to parquet ──
    df_ops = pd.DataFrame(all_operators)
    df_queries = pd.DataFrame(all_queries)

    ops_path = os.path.join(output_dir, 'unified_operators.parquet')
    queries_path = os.path.join(output_dir, 'unified_queries.parquet')

    df_ops.to_parquet(ops_path, index=False)
    df_queries.to_parquet(queries_path, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"ETL Complete!")
    logger.info(f"  Operator instances: {len(df_ops):,}")
    logger.info(f"  Query results: {len(df_queries):,}")
    logger.info(f"  Environments: {df_ops['env_id'].nunique()}")
    logger.info(f"  Operator types: {df_ops['op_type'].nunique()}")
    logger.info(f"  Saved: {ops_path}")
    logger.info(f"  Saved: {queries_path}")

    if errors:
        logger.warning(f"  Errors: {len(errors)}")
        with open(os.path.join(output_dir, 'etl_errors.json'), 'w') as f:
            json.dump(errors, f, indent=2)

    # ── Summary statistics ──
    summary = {
        'total_operator_instances': len(df_ops),
        'total_query_results': len(df_queries),
        'environments': sorted(df_ops['env_id'].unique().tolist()) if len(df_ops) > 0 else [],
        'benchmarks': sorted(df_ops['benchmark'].unique().tolist()) if len(df_ops) > 0 else [],
        'hint_sets': sorted(df_ops['hint_set'].unique().tolist()) if len(df_ops) > 0 else [],
        'operator_types': df_ops['op_type'].value_counts().to_dict() if len(df_ops) > 0 else {},
        'env_counts': df_ops.groupby('env_id').size().to_dict() if len(df_ops) > 0 else {},
        'errors': len(errors),
    }

    with open(os.path.join(output_dir, 'etl_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == '__main__':
    summary = run_etl()
    print(json.dumps(summary, indent=2, ensure_ascii=False))

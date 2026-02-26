# -*- coding: utf-8 -*-
"""
HALO Workload Metadata Registry
=================================

Provides table-level metadata (row counts, sizes, indexes) for benchmarks
so the σ model can understand workload characteristics.

Key Insight:
  A TABLE_SCAN on a 100MB table vs 22GB table behaves radically differently
  under hardware changes. Without this metadata, the model treats both
  identically — which causes poor predictions for new workloads.

Usage:
    from workload_metadata import WORKLOAD_REGISTRY, enrich_operator, register_workload
"""

import math
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Workload Registry
# ═══════════════════════════════════════════════════════════════════════

WORKLOAD_REGISTRY = {
    # ── TPC-H SF30 ──
    'TPCH': {
        'scale_factor': 30,
        'total_size_gb': 30.0,
        'n_tables': 8,
        'tables': {
            # Large fact tables
            'lineitem':  {'rows': 179998994, 'size_mb': 22000,
                          'indexes': ['PRIMARY', 'i_l_orderkey', 'i_l_partkey',
                                      'i_l_suppkey', 'i_l_shipdate',
                                      'i_l_commitdate', 'i_l_receiptdate']},
            'orders':    {'rows': 45000000,  'size_mb': 5500,
                          'indexes': ['PRIMARY', 'i_o_custkey', 'i_o_orderdate']},
            'partsupp':  {'rows': 24000000,  'size_mb': 2800,
                          'indexes': ['PRIMARY', 'i_ps_suppkey']},
            # Medium dimension tables
            'part':      {'rows': 6000000,   'size_mb': 750,
                          'indexes': ['PRIMARY', 'i_p_name', 'i_p_type', 'i_p_size']},
            'customer':  {'rows': 4500000,   'size_mb': 550,
                          'indexes': ['PRIMARY', 'i_c_nationkey']},
            'supplier':  {'rows': 300000,    'size_mb': 40,
                          'indexes': ['PRIMARY', 'i_s_nationkey']},
            # Small lookup tables
            'nation':    {'rows': 25,        'size_mb': 1,
                          'indexes': ['PRIMARY', 'i_n_regionkey']},
            'region':    {'rows': 5,         'size_mb': 1,
                          'indexes': ['PRIMARY']},
        },
        # Alias normalization: TPCH queries use varied case/aliases
        'alias_map': {
            'LINEITEM': 'lineitem', 'ORDERS': 'orders', 'PART': 'part',
            'PARTSUPP': 'partsupp', 'CUSTOMER': 'customer', 'SUPPLIER': 'supplier',
            'NATION': 'nation', 'REGION': 'region',
            'l1': 'lineitem', 'l2': 'lineitem', 'l3': 'lineitem',
            'n1': 'nation', 'n2': 'nation',
            'c_orders': 'orders', 'custsale': None,  # subquery aliases
            'revenue0': None,  # aggregated view
        },
    },

    # ── JOB (Join Order Benchmark) / IMDB  ──
    'JOB': {
        'scale_factor': 1,
        'total_size_gb': 3.6,
        'n_tables': 21,
        'tables': {
            'cast_info':          {'rows': 36244344, 'size_mb': 2300, 'indexes': ['PRIMARY', 'ci_movie_id', 'ci_role_id']},
            'movie_info':         {'rows': 14835720, 'size_mb': 1800, 'indexes': ['PRIMARY', 'mi_movie_id', 'mi_info_type_id']},
            'movie_info_idx':     {'rows': 1380035,  'size_mb': 170,  'indexes': ['PRIMARY', 'miidx_movie_id', 'miidx_info_type_id']},
            'movie_companies':    {'rows': 2609129,  'size_mb': 250,  'indexes': ['PRIMARY', 'mc_movie_id', 'mc_company_id']},
            'movie_keyword':      {'rows': 4523930,  'size_mb': 200,  'indexes': ['PRIMARY', 'mk_movie_id', 'mk_keyword_id']},
            'movie_link':         {'rows': 29997,    'size_mb': 5,    'indexes': ['PRIMARY', 'ml_movie_id']},
            'complete_cast':      {'rows': 135086,   'size_mb': 10,   'indexes': ['PRIMARY', 'cc_movie_id']},
            'title':              {'rows': 2528312,  'size_mb': 400,  'indexes': ['PRIMARY', 't_kind_id', 't_production_year']},
            'aka_title':          {'rows': 361472,   'size_mb': 60,   'indexes': ['PRIMARY', 'at_movie_id']},
            'aka_name':           {'rows': 901343,   'size_mb': 100,  'indexes': ['PRIMARY', 'an_person_id']},
            'char_name':          {'rows': 3140339,  'size_mb': 350,  'indexes': ['PRIMARY']},
            'name':               {'rows': 4167491,  'size_mb': 550,  'indexes': ['PRIMARY']},
            'person_info':        {'rows': 2963664,  'size_mb': 450,  'indexes': ['PRIMARY', 'pi_person_id']},
            'comp_cast_type':     {'rows': 4,        'size_mb': 1,    'indexes': ['PRIMARY']},
            'company_name':       {'rows': 234997,   'size_mb': 25,   'indexes': ['PRIMARY']},
            'company_type':       {'rows': 4,        'size_mb': 1,    'indexes': ['PRIMARY']},
            'info_type':          {'rows': 113,      'size_mb': 1,    'indexes': ['PRIMARY']},
            'keyword':            {'rows': 134170,   'size_mb': 10,   'indexes': ['PRIMARY']},
            'kind_type':          {'rows': 7,        'size_mb': 1,    'indexes': ['PRIMARY']},
            'link_type':          {'rows': 18,       'size_mb': 1,    'indexes': ['PRIMARY']},
            'role_type':          {'rows': 12,       'size_mb': 1,    'indexes': ['PRIMARY']},
        },
        # Alias normalization: JOB queries use short aliases
        'alias_map': {
            'ci': 'cast_info', 'mi': 'movie_info', 'mi_idx': 'movie_info_idx',
            'mi_idx1': 'movie_info_idx', 'mi_idx2': 'movie_info_idx',
            'miidx': 'movie_info_idx',
            'mc': 'movie_companies', 'mc1': 'movie_companies', 'mc2': 'movie_companies',
            'mk': 'movie_keyword', 'ml': 'movie_link',
            'cc': 'complete_cast',
            't': 'title', 't1': 'title', 't2': 'title',
            'at': 'aka_title', 'an': 'aka_name', 'an1': 'aka_name',
            'chn': 'char_name', 'n': 'name', 'n1': 'name',
            'pi': 'person_info',
            'cct1': 'comp_cast_type', 'cct2': 'comp_cast_type',
            'cn': 'company_name', 'cn1': 'company_name', 'cn2': 'company_name',
            'companies': 'company_name',
            'ct': 'company_type',
            'it': 'info_type', 'it1': 'info_type', 'it2': 'info_type', 'it3': 'info_type',
            'k': 'keyword', 'kt': 'kind_type', 'kt1': 'kind_type', 'kt2': 'kind_type',
            'lt': 'link_type', 'rt': 'role_type',
            'Ball': None,   # special alias in some JOB queries
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic Registration
# ═══════════════════════════════════════════════════════════════════════

def register_workload(name: str, metadata: dict):
    """
    Register a new benchmark/workload.

    Args:
        name:     e.g. 'SSB', 'YCSB', 'CustomBench'
        metadata: dict with keys: scale_factor, total_size_gb, n_tables, tables, alias_map
    """
    required = ['total_size_gb', 'tables']
    missing = [k for k in required if k not in metadata]
    if missing:
        raise ValueError(f"Missing required keys for workload '{name}': {missing}")
    WORKLOAD_REGISTRY[name] = metadata
    logger.info(f"Registered workload: {name} ({len(metadata['tables'])} tables, "
                f"{metadata['total_size_gb']:.1f} GB)")


# ═══════════════════════════════════════════════════════════════════════
#  Operator Enrichment
# ═══════════════════════════════════════════════════════════════════════

def _resolve_table_alias(alias: str, benchmark: str) -> str:
    """Resolve a table alias to its canonical table name."""
    wl = WORKLOAD_REGISTRY.get(benchmark)
    if not wl:
        return alias

    # Direct match in tables
    tables = wl.get('tables', {})
    if alias in tables:
        return alias

    # Resolve via alias map
    alias_map = wl.get('alias_map', {})
    # Try exact match
    resolved = alias_map.get(alias)
    if resolved:
        return resolved

    # Try case-insensitive
    alias_lower = alias.lower()
    for k, v in alias_map.items():
        if k.lower() == alias_lower:
            return v

    # Try direct match on table names (case-insensitive)
    for tname in tables:
        if tname.lower() == alias_lower:
            return tname

    return None


def enrich_operator(op: dict, benchmark: str, env_id: str = None) -> dict:
    """
    Enrich an operator dict with workload metadata features.

    Adds:
      - table_rows:       row count of the accessed table
      - table_size_mb:    size in MB of the accessed table
      - n_indexes:        number of indexes on the table
      - dataset_total_gb: total dataset size
      - fits_in_buffer:   1 if table fits in buffer pool, 0 otherwise

    Args:
        op:        operator dict (must have 'table_alias' key)
        benchmark: 'TPCH' or 'JOB'
        env_id:    optional, for buffer pool size lookup
    """
    alias = op.get('table_alias', '')
    table_name = _resolve_table_alias(alias, benchmark) if alias else None

    wl = WORKLOAD_REGISTRY.get(benchmark, {})
    tables = wl.get('tables', {})
    meta = tables.get(table_name, {}) if table_name else {}

    op['table_rows'] = meta.get('rows', 0)
    op['table_size_mb'] = meta.get('size_mb', 0)
    op['n_indexes'] = len(meta.get('indexes', []))
    op['dataset_total_gb'] = wl.get('total_size_gb', 0)
    op['n_tables_in_dataset'] = wl.get('n_tables', 0)

    # Does table fit in buffer pool?
    buffer_pool_gb = 20  # default
    if env_id:
        try:
            from hw_registry import HW_REGISTRY
            buffer_pool_gb = HW_REGISTRY.get(env_id, {}).get('buffer_pool_gb', 20)
        except ImportError:
            pass
    table_size_gb = meta.get('size_mb', 0) / 1024.0
    op['fits_in_buffer'] = int(table_size_gb < buffer_pool_gb)

    return op


def enrich_operators_batch(ops_df, benchmark_col='benchmark', env_col='env_id'):
    """
    Enrich a DataFrame of operators in-place with workload metadata.

    Args:
        ops_df:        DataFrame with operators
        benchmark_col: column name for benchmark
        env_col:       column name for environment
    """
    import pandas as pd

    new_cols = {'table_rows': [], 'table_size_mb': [], 'n_indexes': [],
                'dataset_total_gb': [], 'n_tables_in_dataset': [], 'fits_in_buffer': []}

    for _, row in ops_df.iterrows():
        enriched = enrich_operator(
            row.to_dict(),
            benchmark=row.get(benchmark_col, ''),
            env_id=row.get(env_col, None)
        )
        for k in new_cols:
            new_cols[k].append(enriched[k])

    for k, v in new_cols.items():
        ops_df[k] = v

    logger.info(f"Enriched {len(ops_df):,} operators with workload metadata")
    return ops_df


# ═══════════════════════════════════════════════════════════════════════
#  Quick Demo
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=== HALO Workload Metadata Registry ===\n")

    for bench_name, wl in WORKLOAD_REGISTRY.items():
        print(f"\n--- {bench_name} (SF={wl.get('scale_factor', '?')}, "
              f"{wl['total_size_gb']} GB, {len(wl['tables'])} tables) ---")
        for tname, tmeta in sorted(wl['tables'].items(),
                                    key=lambda x: -x[1]['rows']):
            print(f"  {tname:<22} {tmeta['rows']:>12,} rows  "
                  f"{tmeta['size_mb']:>6,} MB  "
                  f"{len(tmeta['indexes'])} indexes")

    # Test alias resolution
    print("\n--- Alias Resolution Tests ---")
    for alias, bench in [('ci', 'JOB'), ('LINEITEM', 'TPCH'), ('l1', 'TPCH'),
                          ('mi_idx', 'JOB'), ('n1', 'JOB')]:
        resolved = _resolve_table_alias(alias, bench)
        meta = WORKLOAD_REGISTRY[bench]['tables'].get(resolved, {})
        print(f"  {bench}:{alias:<12} → {resolved:<22} "
              f"({meta.get('rows', 0):>12,} rows)")

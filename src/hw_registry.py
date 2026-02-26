# -*- coding: utf-8 -*-
"""
HALO Hardware Registry — Single Source of Truth
================================================

Centralizes all hardware environment specifications with quantitative
performance metrics. Replaces the scattered ENV_PROFILES / ENV_MAP
dictionaries that previously used only binary (0/1) change flags.

Key Design:
  - Each environment stores measurable specs (IOPS, MB/s, cores, GHz, RAM)
  - `compute_hw_features()` generates log-ratio features that tell the
    σ model HOW MUCH the hardware changed, not just WHETHER it changed.
  - `register_env()` allows dynamic addition of new target servers.

Usage:
    from hw_registry import HW_REGISTRY, compute_hw_features, register_env

    hw = compute_hw_features('A_NVMe', 'B_SATA')
    # hw['iops_ratio'] = log(B_SATA_IOPS / A_NVMe_IOPS) → large negative = big downgrade
"""

import os
import math
import platform
import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  Hardware Environment Registry (Quantitative Specs)
# ═══════════════════════════════════════════════════════════════════════

HW_REGISTRY = {
    # ── Server A: Intel Desktop (i9-12900K) ──
    'A_NVMe': {
        'server': 'A',
        'storage': 'NVMe',
        'cpu': 'i9-12900K',
        # Storage specs (Samsung 990 Pro NVMe)
        'seq_read_mbps': 7450,
        'seq_write_mbps': 6900,
        'rand_read_iops': 1200000,
        'rand_write_iops': 1550000,
        # CPU specs
        'cpu_cores': 16,         # 8P + 8E
        'cpu_threads': 24,
        'cpu_base_ghz': 3.2,
        'cpu_boost_ghz': 5.2,
        'l3_cache_mb': 30,
        # Memory
        'ram_gb': 128,
        'buffer_pool_gb': 20,
    },
    'A_SATA': {
        'server': 'A',
        'storage': 'SATA',
        'cpu': 'i9-12900K',
        # Storage specs (WD Blue SATA SSD)
        'seq_read_mbps': 560,
        'seq_write_mbps': 530,
        'rand_read_iops': 95000,
        'rand_write_iops': 84000,
        # CPU specs (same as A_NVMe)
        'cpu_cores': 16,
        'cpu_threads': 24,
        'cpu_base_ghz': 3.2,
        'cpu_boost_ghz': 5.2,
        'l3_cache_mb': 30,
        # Memory
        'ram_gb': 128,
        'buffer_pool_gb': 20,
    },

    # ── Server B: AMD Server (EPYC 7713) ──
    'B_NVMe': {
        'server': 'B',
        'storage': 'NVMe',
        'cpu': 'EPYC-7713',
        # Storage specs (Samsung PM9A3 NVMe)
        'seq_read_mbps': 6900,
        'seq_write_mbps': 4000,
        'rand_read_iops': 1000000,
        'rand_write_iops': 180000,
        # CPU specs
        'cpu_cores': 64,
        'cpu_threads': 128,
        'cpu_base_ghz': 2.0,
        'cpu_boost_ghz': 3.675,
        'l3_cache_mb': 256,
        # Memory
        'ram_gb': 256,
        'buffer_pool_gb': 20,
    },
    'B_SATA': {
        'server': 'B',
        'storage': 'SATA',
        'cpu': 'EPYC-7713',
        # Storage specs (WD Blue SATA SSD)
        'seq_read_mbps': 560,
        'seq_write_mbps': 530,
        'rand_read_iops': 95000,
        'rand_write_iops': 84000,
        # CPU specs (same as B_NVMe)
        'cpu_cores': 64,
        'cpu_threads': 128,
        'cpu_base_ghz': 2.0,
        'cpu_boost_ghz': 3.675,
        'l3_cache_mb': 256,
        # Memory
        'ram_gb': 256,
        'buffer_pool_gb': 20,
    },

    # ── Target Server: Intel Xeon Silver 4310 ──
    'Xeon_NVMe': {
        'server': 'Xeon',
        'storage': 'NVMe',
        'cpu': 'Xeon-4310',
        'seq_read_mbps': 3500,     # typical datacenter NVMe
        'seq_write_mbps': 3000,
        'rand_read_iops': 500000,
        'rand_write_iops': 100000,
        'cpu_cores': 12,
        'cpu_threads': 24,
        'cpu_base_ghz': 2.1,
        'cpu_boost_ghz': 3.3,
        'l3_cache_mb': 18,
        'ram_gb': 64,
        'buffer_pool_gb': 20,
    },
    'Xeon_SATA': {
        'server': 'Xeon',
        'storage': 'SATA',
        'cpu': 'Xeon-4310',
        'seq_read_mbps': 560,
        'seq_write_mbps': 530,
        'rand_read_iops': 95000,
        'rand_write_iops': 84000,
        'cpu_cores': 12,
        'cpu_threads': 24,
        'cpu_base_ghz': 2.1,
        'cpu_boost_ghz': 3.3,
        'l3_cache_mb': 18,
        'ram_gb': 64,
        'buffer_pool_gb': 20,
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic Registration
# ═══════════════════════════════════════════════════════════════════════

_REQUIRED_KEYS = [
    'server', 'storage', 'cpu',
    'seq_read_mbps', 'rand_read_iops',
    'cpu_cores', 'cpu_boost_ghz',
    'ram_gb', 'buffer_pool_gb',
]


def register_env(env_id: str, specs: dict):
    """
    Dynamically register a new hardware environment.

    Args:
        env_id:  e.g. 'C_NVMe'
        specs:   dict with at least the required keys
    """
    missing = [k for k in _REQUIRED_KEYS if k not in specs]
    if missing:
        raise ValueError(f"Missing required specs for {env_id}: {missing}")
    HW_REGISTRY[env_id] = specs
    logger.info(f"Registered new environment: {env_id} ({specs['cpu']}, {specs['storage']})")


def auto_register_local_env(env_id: str, storage_type: str, buffer_pool_gb: float = 20.0):
    """
    Automatically detects local CPU and Memory specs, applies standard I/O 
    averages based on `storage_type` (NVMe, SATA, HDD), and registers it to HW_REGISTRY.
    """
    storage_type = storage_type.upper()
    
    # 1. Detect CPU Cores/Threads
    try:
        cpu_threads = os.cpu_count() or 4
        cpu_cores = max(1, cpu_threads // 2)  # Rough hyperthreading assumption
    except:
        cpu_threads, cpu_cores = (4, 2)

    # 2. Detect CPU Clock & Model (Linux specific)
    base_ghz = 2.0
    boost_ghz = 3.0
    cpu_model = platform.processor() or "Unknown CPU"
    try:
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_model = line.split(':')[1].strip()
                        if 'GHz' in cpu_model:
                            # e.g., "Intel(R) Xeon(R) CPU @ 2.50GHz"
                            base_ghz = float(cpu_model.split('@')[-1].replace('GHz', '').strip())
                            boost_ghz = base_ghz * 1.5 # Conservative boost estimate
                        break
    except:
        pass

    # 3. Detect System Memory (Linux specific)
    ram_gb = 16.0
    try:
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        ram_gb = kb / (1024 * 1024)
                        break
    except:
        pass

    # 4. Apply Default Storage Speeds
    storage_defs = {
        'NVME': {
            'seq_read_mbps': 3500, 'seq_write_mbps': 3000,
            'rand_read_iops': 500000, 'rand_write_iops': 100000,
        },
        'SATA': {
            'seq_read_mbps': 550, 'seq_write_mbps': 500,
            'rand_read_iops': 85000, 'rand_write_iops': 80000,
        },
        'HDD': {
            'seq_read_mbps': 150, 'seq_write_mbps': 150,
            'rand_read_iops': 200, 'rand_write_iops': 200,
        }
    }
    
    if storage_type not in storage_defs:
        logger.warning(f"Unknown storage_type '{storage_type}'. Defaulting to SATA.")
        storage_type = 'SATA'
        
    s_specs = storage_defs[storage_type]

    # Assemble specs
    specs = {
        'server': 'LocalAuto',
        'storage': storage_type,
        'cpu': cpu_model[:30],
        'seq_read_mbps': s_specs['seq_read_mbps'],
        'seq_write_mbps': s_specs['seq_write_mbps'],
        'rand_read_iops': s_specs['rand_read_iops'],
        'rand_write_iops': s_specs['rand_write_iops'],
        'cpu_cores': cpu_cores,
        'cpu_threads': cpu_threads,
        'cpu_base_ghz': round(base_ghz, 2),
        'cpu_boost_ghz': round(boost_ghz, 2),
        'l3_cache_mb': 20,  # Generic placeholder
        'ram_gb': round(ram_gb, 1),
        'buffer_pool_gb': buffer_pool_gb,
    }

    # Register
    register_env(env_id, specs)
    return specs


# ═══════════════════════════════════════════════════════════════════════
#  Quantitative Feature Computation
# ═══════════════════════════════════════════════════════════════════════

def _safe_log_ratio(target_val, source_val):
    """Compute log(target/source). Positive = target is better/bigger."""
    return math.log(max(target_val, 0.001) / max(source_val, 0.001))


def compute_hw_features(source_env: str, target_env: str) -> dict:
    """
    Compute hardware transition features for source → target transfer.

    Returns dict with:
      - Legacy binary flags (storage_changed, compute_changed, both_changed)
      - Quantitative log-ratio features (positive = target is better)
      - Direction flags (is_storage_downgrade, is_cpu_downgrade)

    Example:
        >>> hw = compute_hw_features('A_NVMe', 'B_SATA')
        >>> hw['iops_ratio']       # large negative: NVMe→SATA is a big I/O downgrade
        >>> hw['cpu_core_ratio']   # positive: 16→64 cores is an upgrade
    """
    src = HW_REGISTRY.get(source_env)
    tgt = HW_REGISTRY.get(target_env)

    if src is None or tgt is None:
        missing = source_env if src is None else target_env
        raise KeyError(f"Environment '{missing}' not found in HW_REGISTRY. "
                       f"Available: {list(HW_REGISTRY.keys())}")

    sc = int(src['storage'] != tgt['storage'])
    cc = int(src['cpu'] != tgt['cpu'])

    return {
        # ── Legacy binary flags (backward compatible) ──
        'storage_changed': sc,
        'compute_changed': cc,
        'both_changed': sc * cc,

        # ── Storage quantitative (log-ratio, positive = target faster) ──
        'storage_speed_ratio': _safe_log_ratio(tgt['seq_read_mbps'], src['seq_read_mbps']),
        'iops_ratio': _safe_log_ratio(tgt['rand_read_iops'], src['rand_read_iops']),
        'write_speed_ratio': _safe_log_ratio(tgt['seq_write_mbps'], src['seq_write_mbps']),

        # ── CPU quantitative ──
        'cpu_core_ratio': _safe_log_ratio(tgt['cpu_cores'], src['cpu_cores']),
        'cpu_thread_ratio': _safe_log_ratio(tgt['cpu_threads'], src['cpu_threads']),
        'cpu_clock_ratio': _safe_log_ratio(tgt['cpu_boost_ghz'], src['cpu_boost_ghz']),
        'cpu_base_clock_ratio': _safe_log_ratio(tgt['cpu_base_ghz'], src['cpu_base_ghz']),
        'l3_cache_ratio': _safe_log_ratio(tgt['l3_cache_mb'], src['l3_cache_mb']),

        # ── Memory quantitative ──
        'ram_ratio': _safe_log_ratio(tgt['ram_gb'], src['ram_gb']),
        'buffer_pool_ratio': _safe_log_ratio(tgt['buffer_pool_gb'], src['buffer_pool_gb']),

        # ── Direction flags (explicit downgrade signal) ──
        'is_storage_downgrade': int(tgt['rand_read_iops'] < src['rand_read_iops']),
        'is_cpu_downgrade': int(tgt['cpu_boost_ghz'] < src['cpu_boost_ghz']),
        'is_ram_downgrade': int(tgt['ram_gb'] < src['ram_gb']),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Utility: List all environment pairs
# ═══════════════════════════════════════════════════════════════════════

def get_env_pairs(include_target_envs=False):
    """
    Generate all valid training environment pairs.

    Args:
        include_target_envs: if True, include Xeon_* environments.
                             Default False (training only on A/B servers).
    """
    env_ids = sorted(HW_REGISTRY.keys())
    if not include_target_envs:
        env_ids = [e for e in env_ids if not e.startswith('Xeon')]

    pairs = []
    for i, e1 in enumerate(env_ids):
        for e2 in env_ids[i+1:]:
            hw = compute_hw_features(e1, e2)
            hw['env1'] = e1
            hw['env2'] = e2
            hw['pair_label'] = f"{e1}_vs_{e2}"
            pairs.append(hw)
    return pairs


# ═══════════════════════════════════════════════════════════════════════
#  Quick Demo
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=== HALO Hardware Registry ===\n")
    print(f"Registered environments: {list(HW_REGISTRY.keys())}\n")

    # Show a transfer example
    for src, tgt in [('A_NVMe', 'B_SATA'), ('A_NVMe', 'Xeon_SATA')]:
        hw = compute_hw_features(src, tgt)
        print(f"\n--- {src} → {tgt} ---")
        for k, v in hw.items():
            if isinstance(v, float):
                print(f"  {k:<25} {v:>+8.3f}")
            else:
                print(f"  {k:<25} {v}")

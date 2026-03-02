import re
import sys

def parse_progress_file(filepath):
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            # Match lines like: [1/113] 1a.sql - ✓ SUCCESS - 1.836252794s
            match = re.search(r'\[\d+/\d+\] ([\w\.]+sql) - ✓ SUCCESS - ([\d\.]+)s', line)
            if match:
                query_name = match.group(1)
                time_s = float(match.group(2))
                results[query_name] = time_s
    return results

def get_stats(orig_file, halo_file):
    orig_results = parse_progress_file(orig_file)
    halo_results = parse_progress_file(halo_file)
    
    if not orig_results or not halo_results:
        return None
        
    orig_total = sum(orig_results.values())
    halo_total = sum(halo_results.values())
    
    speedups = {}
    for q in orig_results:
        if q in halo_results:
            speedups[q] = orig_results[q] / halo_results[q]
            
    halo_faster = len([q for q, s in speedups.items() if s > 1.05])
    halo_slower = len([q for q, s in speedups.items() if s < 0.95])
    neutral = len([q for q, s in speedups.items() if 0.95 <= s <= 1.05])
    
    return {
        'orig_total': orig_total,
        'halo_total': halo_total,
        'speedup': orig_total / halo_total,
        'time_reduction': (orig_total - halo_total) / orig_total * 100,
        'faster': halo_faster,
        'slower': halo_slower,
        'neutral': neutral,
        'speedups': speedups,
        'orig_results': orig_results,
        'halo_results': halo_results
    }

def main():
    nvme_halo_file = '/root/halo/xeon_results/JOB/NVMe/halo/progress.txt'
    nvme_orig_file = '/root/halo/xeon_results/JOB/NVMe/original/progress.txt'
    
    sata_halo_file = '/root/halo/xeon_results/JOB/SATA/halo/progress.txt'
    sata_orig_file = '/root/halo/xeon_results/JOB/SATA/original/progress.txt'

    nvme_stats = get_stats(nvme_orig_file, nvme_halo_file)
    sata_stats = get_stats(sata_orig_file, sata_halo_file)

    if not nvme_stats or not sata_stats:
        print("Failed to parse log files!")
        return

    print("=== SATA Environment Summary ===")
    print(f"Original Time: {sata_stats['orig_total']:.2f}s")
    print(f"HALO Time: {sata_stats['halo_total']:.2f}s")
    print(f"Overall Speedup: {sata_stats['speedup']:.2f}x")
    print(f"Time Reduction: {sata_stats['time_reduction']:.2f}%")
    print(f"Queries Improved (>1.05x): {sata_stats['faster']}")
    print(f"Queries Regressed (<0.95x): {sata_stats['slower']}")
    
    print("\n=== NVMe Environment Summary ===")
    print(f"Original Time: {nvme_stats['orig_total']:.2f}s")
    print(f"HALO Time: {nvme_stats['halo_total']:.2f}s")
    print(f"Overall Speedup: {nvme_stats['speedup']:.2f}x")
    print(f"Time Reduction: {nvme_stats['time_reduction']:.2f}%")
    print(f"Queries Improved (>1.05x): {nvme_stats['faster']}")
    print(f"Queries Regressed (<0.95x): {nvme_stats['slower']}")
    
    print("\n=== Direct Comparison (SATA vs NVMe) ===")
    print(f"Base Original Time diff (SATA -> NVMe): {sata_stats['orig_total']:.2f}s -> {nvme_stats['orig_total']:.2f}s ({(sata_stats['orig_total']/nvme_stats['orig_total']):.2f}x faster hardware baseline)")
    print(f"HALO Speedup Diff (SATA vs NVMe): {sata_stats['speedup']:.2f}x vs {nvme_stats['speedup']:.2f}x")
    
    print("\nTop 5 Absolute Time Reductions (SATA):")
    sata_time_saved = {q: sata_stats['orig_results'][q] - sata_stats['halo_results'][q] for q in sata_stats['orig_results']}
    sata_sorted_saved = sorted(sata_time_saved.items(), key=lambda x: x[1], reverse=True)
    for q, saved in sata_sorted_saved[:5]:
        print(f"  {q}: saved {saved:.2f}s (Orig: {sata_stats['orig_results'][q]:.2f}s, HALO: {sata_stats['halo_results'][q]:.2f}s, Speedup: {sata_stats['speedups'][q]:.2f}x)")

    print("\nTop 5 Absolute Time Reductions (NVMe):")
    nvme_time_saved = {q: nvme_stats['orig_results'][q] - nvme_stats['halo_results'][q] for q in nvme_stats['orig_results']}
    nvme_sorted_saved = sorted(nvme_time_saved.items(), key=lambda x: x[1], reverse=True)
    for q, saved in nvme_sorted_saved[:5]:
        print(f"  {q}: saved {saved:.2f}s (Orig: {nvme_stats['orig_results'][q]:.2f}s, HALO: {nvme_stats['halo_results'][q]:.2f}s, Speedup: {nvme_stats['speedups'][q]:.2f}x)")

if __name__ == '__main__':
    main()

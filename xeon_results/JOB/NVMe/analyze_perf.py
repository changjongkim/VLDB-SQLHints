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

def main():
    halo_file = '/root/halo/xeon_results/JOB/NVMe/halo/progress.txt'
    orig_file = '/root/halo/xeon_results/JOB/NVMe/original/progress.txt'

    halo_results = parse_progress_file(halo_file)
    orig_results = parse_progress_file(orig_file)

    if not halo_results or not orig_results:
        print("Failed to parse log files!")
        return

    halo_total = sum(halo_results.values())
    orig_total = sum(orig_results.values())

    print(f"Total Queries: {len(halo_results)}")
    print(f"Original Total Time: {orig_total:.2f}s")
    print(f"HALO Total Time: {halo_total:.2f}s")
    print(f"Total Speedup: {orig_total / halo_total:.2f}x")
    print(f"Total Time Reduction: {(orig_total - halo_total) / orig_total * 100:.2f}%")

    speedups = {}
    for q in orig_results:
        if q in halo_results:
            speedup = orig_results[q] / halo_results[q]
            speedups[q] = speedup

    sorted_speedups = sorted(speedups.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Query Speedups:")
    for i in range(10):
        if i < len(sorted_speedups):
            q, s = sorted_speedups[i]
            print(f"  {q}: {s:.2f}x (Orig: {orig_results[q]:.2f}s, HALO: {halo_results[q]:.2f}s)")

    print("\nWorst 5 Query Speedups (Regressions):")
    for i in range(1, 6):
        if i <= len(sorted_speedups):
            q, s = sorted_speedups[-i]
            print(f"  {q}: {s:.2f}x (Orig: {orig_results[q]:.2f}s, HALO: {halo_results[q]:.2f}s)")

    halo_faster = len([q for q, s in speedups.items() if s > 1.05])
    halo_slower = len([q for q, s in speedups.items() if s < 0.95])
    neutral = len([q for q, s in speedups.items() if 0.95 <= s <= 1.05])

    print(f"\nSignificantly Faster (>1.05x): {halo_faster} queries")
    print(f"Significantly Slower (<0.95x): {halo_slower} queries")
    print(f"Neutral (0.95x - 1.05x): {neutral} queries")

if __name__ == '__main__':
    main()

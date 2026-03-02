import json
import re

def parse_progress(filepath):
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'\[\d+/\d+\] ([\w\.]+sql) - ✓ SUCCESS - ([\d\.]+)s', line)
            if match:
                q = match.group(1).replace('.sql', '')
                t = float(match.group(2))
                results[q] = t
    return results

def main():
    orig_path = '/root/halo/xeon_results/JOB/NVMe/original/progress.txt'
    halo_path = '/root/halo/xeon_results/JOB/NVMe/halo/progress.txt'
    summary_path = '/root/halo/results/hinted_queries_xeon_v4/recommendation_summary.json'

    # 1. Parse execution times
    orig_times = parse_progress(orig_path)
    halo_times = parse_progress(halo_path)

    # 2. Parse recommendations
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)

    # Build map for NVMe JOB
    rec_map = {}
    for item in summary_data:
        if item.get('scenario') == 'Xeon_NVMe' and item.get('benchmark') == 'JOB':
            rec_map[item['query_id']] = item

    # 3. Correlate
    correlation = []
    for q in orig_times:
        if q in halo_times and q in rec_map:
            orig_t = orig_times[q]
            halo_t = halo_times[q]
            speedup = orig_t / halo_t
            rec = rec_map[q]
            correlation.append({
                'query': q,
                'orig_t': orig_t,
                'halo_t': halo_t,
                'speedup': speedup,
                'risk': rec['risk_level'],
                'hint': rec['recommended_hint'],
                'reason': rec['reason']
            })

    # Analysis 1: Regressions
    print("### Analysis of Regressions (Speedup < 0.95)")
    regressions = [c for c in correlation if c['speedup'] < 0.95]
    regressions.sort(key=lambda x: x['speedup'])
    for r in regressions:
        print(f"Query: {r['query']} | Speedup: {r['speedup']:.2f}x | Risk: {r['risk']} | Hint: {r['hint']} | Orig: {r['orig_t']:.3f}s, Halo: {r['halo_t']:.3f}s")

    # Analysis 2: Hits vs Risk
    print("\n### Performance by Risk Level")
    risks = ['GREEN', 'YELLOW', 'ORANGE', 'SAFE']
    for risk in risks:
        subset = [c for c in correlation if c['risk'] == risk]
        if not subset: continue
        avg_speedup = sum(c['speedup'] for c in subset) / len(subset)
        total_saved = sum(c['orig_t'] - c['halo_t'] for c in subset)
        print(f"Risk: {risk:6} | Count: {len(subset):3} | Avg Speedup: {avg_speedup:.2f}x | Total Saved: {total_saved:8.2f}s")

    # Check 11d specifically
    d11 = next((c for c in correlation if c['query'] == '11d'), None)
    if d11:
        print(f"\n### Special Check: 11d")
        print(f"11d: {d11}")

if __name__ == '__main__':
    main()

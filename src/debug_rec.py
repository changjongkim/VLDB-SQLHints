from halo_framework import get_halo_v4
import logging

logging.basicConfig(level=logging.INFO)

halo = get_halo_v4()
qid = 'tpch_q9'
# Source A_NVMe -> Target Xeon_NVMe
rec = halo.recommend(qid, 'A_NVMe', 'Xeon_NVMe', policy='robust')

print(f"\nFinal Recommendation: {rec.recommended_hint}")
print(f"Status: {rec.risk_level}")
print(f"Reason: {rec.reason}")

if rec.details:
    print("\nCandidate details:")
    for c in rec.details:
        print(f"  Hint: {c.hint_set}")
        print(f"    Source Speedup: {c.source_speedup:.2f}")
        print(f"    Expected Gain: {c.expected_gain:.2f}")
        print(f"    Total Risk Score: {c.total_risk_score:.2f}")
        print(f"    N High Risk: {c.n_high_risk}")
        # Print first few operator details
        for i, op in enumerate(c.operator_details[:3]):
            print(f"      Op {i}: {op.op_type} | mu={op.mu:.3f} | sigma={op.sigma:.3f} | upper={op.upper_bound:.3f} | thresh={halo._get_adaptive_threshold(op.op_type)}")

-- HALO Optimized
-- Query: tpch_q14
-- Scenario: Xeon_NVMe
-- Reason: HALO-U (Novelty): 'hint02' (score=0.02, src_sp=1.3x, risk=0%) from B_SATA
-- using default substitutions


select /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */
	100.00 * sum(case
		when p_type like 'PROMO%'
			then l_extendedprice * (1 - l_discount)
		else 0
	end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
	lineitem,
	part
where
	l_partkey = p_partkey
	and l_shipdate >= date '1995-09-01'
	and l_shipdate < date '1995-09-01' + interval '1' month;
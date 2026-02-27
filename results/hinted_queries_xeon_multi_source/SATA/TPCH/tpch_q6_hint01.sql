-- HALO Optimized
-- Query: tpch_q6
-- Scenario: Xeon_SATA
-- Reason: HALO-U (Novelty): 'hint01' (score=0.00, src_sp=1.1x, risk=0%) from B_SATA
-- using default substitutions


select /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1994-01-01'
	and l_shipdate < date '1994-01-01' + interval '1' year
	and l_discount between .06 - 0.01 and .06 + 0.01
	and l_quantity < 24;
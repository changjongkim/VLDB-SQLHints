-- HALO Optimized
-- Query: tpch_q17
-- Scenario: Xeon_SATA
-- Reason: HALO-U (Novelty): 'hint05' (score=0.00, src_sp=1.1x, risk=15%) from A_NVMe
-- using default substitutions


select /*+ SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY) */
	sum(l_extendedprice) / 7.0 as avg_yearly
from
	lineitem,
	part
where
	p_partkey = l_partkey
	and p_brand = 'Brand#23'
	and p_container = 'MED BOX'
	and l_quantity < (
		select
			0.2 * avg(l_quantity)
		from
			lineitem
		where
			l_partkey = p_partkey
	);
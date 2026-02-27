-- HALO Optimized
-- Query: tpch_q5
-- Scenario: Xeon_SATA
-- Reason: HALO-U (Novelty): 'hint04' (score=0.00, src_sp=2.2x, risk=12%) from A_NVMe
-- using default substitutions


select /*+ SET_VAR(optimizer_switch="block_nested_loop=off,hash_join=on") */
	n_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue
from
	customer,
	orders,
	lineitem,
	supplier,
	nation,
	region
where
	c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and l_suppkey = s_suppkey
	and c_nationkey = s_nationkey
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = 'ASIA'
	and o_orderdate >= date '1994-01-01'
	and o_orderdate < date '1994-01-01' + interval '1' year
group by
	n_name
order by
	revenue desc;
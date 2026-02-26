-- HALO Recommended SQL
-- Query     : tpch_q5 (TPCH)
-- Scenario  : A_NVMe → Xeon_NVMe (AMD EPYC Target)
-- Hint      : hint04
-- Reason    : HALO-R: 'hint04' selected (src=2.16x, risk_ops=12/48, risk_ratio=25%)
-- Hint Str  : SET_VAR(optimizer_switch="block_nested_loop=off,hash_join=on")
--

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
	revenue desc;;

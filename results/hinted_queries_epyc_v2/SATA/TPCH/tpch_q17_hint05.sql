-- HALO Recommended SQL
-- Query     : tpch_q17 (TPCH)
-- Scenario  : A_NVMe → B_SATA (AMD EPYC Target)
-- Hint      : hint05
-- Reason    : HALO-R: 'hint05' selected cautiously (src=1.05x, risk_ratio=38% HIGH)
-- Hint Str  : SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY)
--

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
	);;

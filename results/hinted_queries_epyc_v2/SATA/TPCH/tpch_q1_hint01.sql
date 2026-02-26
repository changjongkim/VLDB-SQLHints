-- HALO Recommended SQL
-- Query     : tpch_q1 (TPCH)
-- Scenario  : A_NVMe → B_SATA (AMD EPYC Target)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' selected cautiously (src=1.01x, risk_ratio=50% HIGH)
-- Hint Str  : SET_VAR(optimizer_switch="block_nested_loop=off")
--

select /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */
	l_returnflag,
	l_linestatus,
	sum(l_quantity) as sum_qty,
	sum(l_extendedprice) as sum_base_price,
	sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
	sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
	avg(l_quantity) as avg_qty,
	avg(l_extendedprice) as avg_price,
	avg(l_discount) as avg_disc,
	count(*) as count_order
from
	lineitem
where
	l_shipdate <= DATE '1998-12-01' - INTERVAL 90 DAY
group by
	l_returnflag,
	l_linestatus
order by
	l_returnflag,
	l_linestatus;;

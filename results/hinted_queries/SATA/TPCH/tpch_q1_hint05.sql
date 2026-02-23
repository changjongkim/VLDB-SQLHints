-- HALO Recommended SQL
-- Query     : tpch_q1 (TPCH)
-- Scenario  : A_NVMe → Xeon_SATA (Intel Xeon Silver 4310)
-- Hint      : hint05
-- Reason    : HALO-R: 'hint05' selected (src=1.02x, risk_ops=0/12, risk_ratio=0%)
-- Hint Str  : SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY)
--

select /*+ SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY) */
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

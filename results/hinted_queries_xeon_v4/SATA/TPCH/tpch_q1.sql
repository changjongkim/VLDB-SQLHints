-- HALO v4 Recommended SQL (Performance-Focused)
-- Query     : tpch_q1
-- Scenario  : Xeon_SATA
-- Mode      : HALO-P v4 (Power/Performance Mode)
-- Hint      : hint05
-- Risk Level : GREEN
-- Reason    : Performance candidate selected (Gain=1.02)
======================================================================

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
	l_linestatus;

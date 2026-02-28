-- HALO v4 Recommended SQL (Performance-Focused)
-- Query     : tpch_q11
-- Scenario  : Xeon_NVMe
-- Mode      : HALO-P v4 (Power/Performance Mode)
-- Hint      : hint01
-- Risk Level : ORANGE
-- Reason    : Performance candidate selected (Gain=1.98)
======================================================================

select /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */
	ps_partkey,
	sum(ps_supplycost * ps_availqty) as value
from
	partsupp,
	supplier,
	nation
where
	ps_suppkey = s_suppkey
	and s_nationkey = n_nationkey
	and n_name = 'GERMANY'
group by
	ps_partkey having
		sum(ps_supplycost * ps_availqty) > (
			select
				sum(ps_supplycost * ps_availqty) * 0.0001000000
			from
				partsupp,
				supplier,
				nation
			where
				ps_suppkey = s_suppkey
				and s_nationkey = n_nationkey
				and n_name = 'GERMANY'
		)
order by
	value desc;

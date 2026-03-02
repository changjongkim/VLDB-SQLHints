-- HALO v4 Recommended SQL (Performance-Focused)
-- Query     : tpch_q17
-- Scenario  : Xeon_SATA
-- Mode      : HALO-P v4 (Power/Performance Mode)
-- Hint      : hint05
-- Risk Level : ORANGE
-- Reason    : Performance candidate selected (Gain=1.04)
======================================================================

select
	sum(l_extendedprice) / 7.0 as avg_yearly
from
	lineitem,
	part IGNORE INDEX (PRIMARY)
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

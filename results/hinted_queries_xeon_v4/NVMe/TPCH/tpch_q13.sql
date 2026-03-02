-- HALO v4 Recommended SQL (Performance-Focused)
-- Query     : tpch_q13
-- Scenario  : Xeon_NVMe
-- Mode      : HALO-P v4 (Power/Performance Mode)
-- Hint      : hint05
-- Risk Level : ORANGE
-- Reason    : Performance candidate selected (Gain=2.88)
======================================================================

select
	c_count,
	count(*) as custdist
from
	(
		select
			c_custkey,
			count(o_orderkey)
		from
			customer left outer join orders FORCE INDEX (PRIMARY) on
				c_custkey = o_custkey
				and o_comment not like '%special%requests%'
		group by
			c_custkey
	) as c_orders (c_custkey, c_count)
group by
	c_count
order by
	custdist desc,
	c_count desc;

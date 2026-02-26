-- HALO Recommended SQL
-- Query     : tpch_q13 (TPCH)
-- Target HW : Xeon_NVMe
-- Mode      : Multi-Source Global Selection
-- Hint      : hint01
-- From Src  : B_NVMe
-- Reason    : HALO-R (Multi-Source): 'hint01' selected from B_NVMe (src_speedup=1.08x, risk=30%)
-- Generated : 2026-02-26
======================================================================

select /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */
	c_count,
	count(*) as custdist
from
	(
		select
			c_custkey,
			count(o_orderkey)
		from
			customer left outer join orders on
				c_custkey = o_custkey
				and o_comment not like '%special%requests%'
		group by
			c_custkey
	) as c_orders (c_custkey, c_count)
group by
	c_count
order by
	custdist desc,
	c_count desc;;

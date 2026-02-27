-- HALO Recommended SQL
-- Query     : tpch_q13 (TPCH)
-- Target HW : Xeon_NVMe
-- Mode      : Multi-Source Global Selection
-- Hint      : hint02
-- From Src  : B_SATA
-- Reason    : HALO-U (Novelty): 'hint02' selected (score=0.00, src_sp=1.06x, risk=10%)
-- Generated : 2026-02-26
======================================================================

select /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */
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

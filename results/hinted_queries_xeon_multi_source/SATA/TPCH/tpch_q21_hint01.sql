-- HALO Recommended SQL
-- Query     : tpch_q21 (TPCH)
-- Target HW : Xeon_SATA
-- Mode      : Multi-Source Global Selection
-- Hint      : hint01
-- From Src  : A_SATA
-- Reason    : HALO-R (Multi-Source): 'hint01' selected from A_SATA (src_speedup=1.52x, risk=11%)
-- Generated : 2026-02-26
======================================================================

select /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */
	s_name,
	count(*) as numwait
from
	supplier,
	lineitem l1,
	orders,
	nation
where
	s_suppkey = l1.l_suppkey
	and o_orderkey = l1.l_orderkey
	and o_orderstatus = 'F'
	and l1.l_receiptdate > l1.l_commitdate
	and exists (
		select
			*
		from
			lineitem l2
		where
			l2.l_orderkey = l1.l_orderkey
			and l2.l_suppkey <> l1.l_suppkey
	)
	and not exists (
		select
			*
		from
			lineitem l3
		where
			l3.l_orderkey = l1.l_orderkey
			and l3.l_suppkey <> l1.l_suppkey
			and l3.l_receiptdate > l3.l_commitdate
	)
	and s_nationkey = n_nationkey
	and n_name = 'SAUDI ARABIA'
group by
	s_name
order by
	numwait desc,
	s_name LIMIT 100;;

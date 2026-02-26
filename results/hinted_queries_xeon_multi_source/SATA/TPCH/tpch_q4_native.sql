-- HALO Recommended SQL
-- Query     : tpch_q4 (TPCH)
-- Target HW : Xeon_SATA
-- Mode      : Multi-Source Global Selection
-- Hint      : NATIVE (no hint)
-- From Src  : N/A
-- Reason    : HALO-R (Multi-Source): No safe improving hints found across all known servers. -> NATIVE
-- Generated : 2026-02-26
======================================================================

select
	o_orderpriority,
	count(*) as order_count
from
	orders
where
	o_orderdate >= date '1993-07-01'
	and o_orderdate < date '1993-07-01' + interval '3' month
	and exists (
		select
			*
		from
			lineitem
		where
			l_orderkey = o_orderkey
			and l_commitdate < l_receiptdate
	)
group by
	o_orderpriority
order by
	o_orderpriority;;

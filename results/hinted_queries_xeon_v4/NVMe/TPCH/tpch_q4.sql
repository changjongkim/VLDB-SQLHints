-- HALO v4 Recommended SQL (Performance-Focused)
-- Query     : tpch_q4
-- Scenario  : Xeon_NVMe
-- Mode      : HALO-P v4 (Power/Performance Mode)
-- Hint      : NATIVE
-- Risk Level : SAFE
-- Reason    : Fallback: Low gain 0.91
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
	o_orderpriority;

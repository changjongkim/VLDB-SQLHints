-- HALO Recommended SQL
-- Query     : tpch_q4 (TPCH)
-- Scenario  : A_NVMe → B_SATA (AMD EPYC Target)
-- Hint      : NATIVE (no hint)
-- Reason    : HALO-R: All hints too risky. hint01(53%), hint02(52%), hint04(53%), hint05(55%) → NATIVE
-- Hint Str  : N/A
--

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

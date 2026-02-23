-- HALO Recommended SQL
-- Query     : tpch_q12 (TPCH)
-- Scenario  : A_NVMe → Xeon_NVMe (Intel Xeon Silver 4310)
-- Hint      : NATIVE (no hint)
-- Reason    : HALO-R: All hints rejected. hint01(risk=7), hint02(risk=3), hint05(risk=5) → NATIVE
-- Hint Str  : N/A
--

select
	l_shipmode,
	sum(case
		when o_orderpriority = '1-URGENT'
			or o_orderpriority = '2-HIGH'
			then 1
		else 0
	end) as high_line_count,
	sum(case
		when o_orderpriority <> '1-URGENT'
			and o_orderpriority <> '2-HIGH'
			then 1
		else 0
	end) as low_line_count
from
	orders,
	lineitem
where
	o_orderkey = l_orderkey
	and l_shipmode in ('MAIL', 'SHIP')
	and l_commitdate < l_receiptdate
	and l_shipdate < l_commitdate
	and l_receiptdate >= date '1994-01-01'
	and l_receiptdate < date '1994-01-01' + interval '1' year
group by
	l_shipmode
order by
	l_shipmode;;

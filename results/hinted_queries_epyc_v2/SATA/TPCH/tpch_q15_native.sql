-- HALO Recommended SQL
-- Query     : tpch_q15 (TPCH)
-- Scenario  : A_NVMe → B_SATA (AMD EPYC Target)
-- Hint      : NATIVE (no hint)
-- Reason    : HALO-R: All hints too risky. hint01(69%), hint02(69%), hint04(69%), hint05(69%) → NATIVE
-- Hint Str  : N/A
--

WITH revenue0 (supplier_no, total_revenue) AS (
	select
		l_suppkey,
		sum(l_extendedprice * (1 - l_discount))
	from
		lineitem
	where
		l_shipdate >= date '1996-01-01'
		and l_shipdate < date '1996-01-01' + interval '3' month
	group by
		l_suppkey
)
select
	s_suppkey,
	s_name,
	s_address,
	s_phone,
	total_revenue
from
	supplier,
	revenue0
where
	s_suppkey = supplier_no
	and total_revenue = (
		select
			max(total_revenue)
		from
			revenue0
	)
order by
	s_suppkey;;

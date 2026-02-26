-- HALO Recommended SQL
-- Query     : tpch_q15 (TPCH)
-- Scenario  : A_NVMe → Xeon_SATA (AMD EPYC Target)
-- Hint      : hint05
-- Reason    : HALO-R: 'hint05' selected (src=1.00x, risk_ops=9/39, risk_ratio=23%)
-- Hint Str  : SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY)
--

WITH revenue0 (supplier_no, total_revenue) AS (
	select /*+ SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY) */
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

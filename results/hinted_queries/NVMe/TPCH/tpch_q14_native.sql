-- HALO Recommended SQL
-- Query     : tpch_q14 (TPCH)
-- Scenario  : A_NVMe → Xeon_NVMe (Intel Xeon Silver 4310)
-- Hint      : NATIVE (no hint)
-- Reason    : HALO-R: All hints rejected. hint01(risk=7), hint02(risk=3), hint05(risk=2) → NATIVE
-- Hint Str  : N/A
--

select
	100.00 * sum(case
		when p_type like 'PROMO%'
			then l_extendedprice * (1 - l_discount)
		else 0
	end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
	lineitem,
	part
where
	l_partkey = p_partkey
	and l_shipdate >= date '1995-09-01'
	and l_shipdate < date '1995-09-01' + interval '1' month;;

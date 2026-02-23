-- HALO Recommended SQL
-- Query     : tpch_q2 (TPCH)
-- Scenario  : A_NVMe → Xeon_NVMe (Intel Xeon Silver 4310)
-- Hint      : NATIVE (no hint)
-- Reason    : HALO-R: All hints too risky. hint01(65%), hint02(70%), hint04(56%), hint05(65%) → NATIVE
-- Hint Str  : N/A
--

select
	s_acctbal,
	s_name,
	n_name,
	p_partkey,
	p_mfgr,
	s_address,
	s_phone,
	s_comment
from
	part,
	supplier,
	partsupp,
	nation,
	region
where
	p_partkey = ps_partkey
	and s_suppkey = ps_suppkey
	and p_size = 15
	and p_type like '%BRASS'
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = 'EUROPE'
	and ps_supplycost = (
		select
			min(ps_supplycost)
		from
			partsupp,
			supplier,
			nation,
			region
		where
			p_partkey = ps_partkey
			and s_suppkey = ps_suppkey
			and s_nationkey = n_nationkey
			and n_regionkey = r_regionkey
			and r_name = 'EUROPE'
	)
order by
	s_acctbal desc,
	n_name,
	s_name,
	p_partkey LIMIT 100;;

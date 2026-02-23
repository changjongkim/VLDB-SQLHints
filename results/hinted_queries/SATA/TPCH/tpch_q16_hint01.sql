-- HALO Recommended SQL
-- Query     : tpch_q16 (TPCH)
-- Scenario  : A_NVMe → Xeon_SATA (Intel Xeon Silver 4310)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' selected cautiously (src=1.41x, risk_ratio=38% HIGH)
-- Hint Str  : SET_VAR(optimizer_switch="block_nested_loop=off")
--

select /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */
	p_brand,
	p_type,
	p_size,
	count(distinct ps_suppkey) as supplier_cnt
from
	partsupp,
	part
where
	p_partkey = ps_partkey
	and p_brand <> 'Brand#45'
	and p_type not like 'MEDIUM POLISHED%'
	and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
	and ps_suppkey not in (
		select
			s_suppkey
		from
			supplier
		where
			s_comment like '%Customer%Complaints%'
	)
group by
	p_brand,
	p_type,
	p_size
order by
	supplier_cnt desc,
	p_brand,
	p_type,
	p_size;;

-- HALO Recommended SQL
-- Query     : tpch_q5 (TPCH)
-- Target HW : Xeon_SATA
-- Mode      : Multi-Source Global Selection
-- Hint      : NATIVE (no hint)
-- From Src  : N/A
-- Reason    : HALO-R (Multi-Source): No safe improving hints found across all known servers. -> NATIVE
-- Generated : 2026-02-26
======================================================================

select
	n_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue
from
	customer,
	orders,
	lineitem,
	supplier,
	nation,
	region
where
	c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and l_suppkey = s_suppkey
	and c_nationkey = s_nationkey
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = 'ASIA'
	and o_orderdate >= date '1994-01-01'
	and o_orderdate < date '1994-01-01' + interval '1' year
group by
	n_name
order by
	revenue desc;;

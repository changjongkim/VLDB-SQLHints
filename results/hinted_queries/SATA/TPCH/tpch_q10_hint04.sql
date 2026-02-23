-- HALO Recommended SQL
-- Query     : tpch_q10 (TPCH)
-- Scenario  : A_NVMe → Xeon_SATA (Intel Xeon Silver 4310)
-- Hint      : hint04
-- Reason    : HALO-R: 'hint04' selected (src=1.00x, risk_ops=7/38, risk_ratio=18%)
-- Hint Str  : SET_VAR(optimizer_switch="block_nested_loop=off,hash_join=on")
--

select /*+ SET_VAR(optimizer_switch="block_nested_loop=off,hash_join=on") */
	c_custkey,
	c_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue,
	c_acctbal,
	n_name,
	c_address,
	c_phone,
	c_comment
from
	customer,
	orders,
	lineitem,
	nation
where
	c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and o_orderdate >= date '1993-10-01'
	and o_orderdate < date '1993-10-01' + interval '3' month
	and l_returnflag = 'R'
	and c_nationkey = n_nationkey
group by
	c_custkey,
	c_name,
	c_acctbal,
	c_phone,
	n_name,
	c_address,
	c_comment
order by
	revenue desc LIMIT 20;;

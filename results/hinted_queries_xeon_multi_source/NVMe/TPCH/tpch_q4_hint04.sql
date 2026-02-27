-- HALO Optimized
-- Query: tpch_q4
-- Scenario: Xeon_NVMe
-- Reason: HALO-U (Novelty): 'hint04' (score=0.00, src_sp=1.1x, risk=27%) from A_SATA
-- using default substitutions


select /*+ SET_VAR(optimizer_switch="block_nested_loop=off,hash_join=on") */
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
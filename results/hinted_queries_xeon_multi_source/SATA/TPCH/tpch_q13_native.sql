-- HALO Optimized
-- Query: tpch_q13
-- Scenario: Xeon_SATA
-- Reason: HALO-U: Uncertainty too high (Gate=15%)
-- using default substitutions


select
	c_count,
	count(*) as custdist
from
	(
		select
			c_custkey,
			count(o_orderkey)
		from
			customer left outer join orders on
				c_custkey = o_custkey
				and o_comment not like '%special%requests%'
		group by
			c_custkey
	) as c_orders (c_custkey, c_count)
group by
	c_count
order by
	custdist desc,
	c_count desc;


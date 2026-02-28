-- HALO v4 Recommended SQL (Performance-Focused)
-- Query     : tpch_q6
-- Scenario  : Xeon_SATA
-- Mode      : HALO-P v4 (Power/Performance Mode)
-- Hint      : hint02
-- Risk Level : GREEN
-- Reason    : Performance candidate selected (Gain=1.01)
======================================================================

select /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1994-01-01'
	and l_shipdate < date '1994-01-01' + interval '1' year
	and l_discount between .06 - 0.01 and .06 + 0.01
	and l_quantity < 24;

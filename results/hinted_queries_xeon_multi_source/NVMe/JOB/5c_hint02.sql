-- HALO Recommended SQL
-- Query     : 5c (JOB)
-- Target HW : Xeon_NVMe
-- Mode      : Multi-Source Global Selection
-- Hint      : hint02
-- From Src  : A_SATA
-- Reason    : HALO-R (Multi-Source): 'hint02' selected from A_SATA (src_speedup=1.56x, risk=27%)
-- Generated : 2026-02-26
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */ MIN(t.title) AS american_movie FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info AS mi, title AS t WHERE ct.kind  = 'production companies' AND mc.note  not like '%(TV)%' and mc.note like '%(USA)%' AND mi.info  IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 1990 AND t.id = mi.movie_id AND t.id = mc.movie_id AND mc.movie_id = mi.movie_id AND ct.id = mc.company_type_id AND it.id = mi.info_type_id;;

-- HALO Optimized
-- Query: 5c
-- Scenario: Xeon_SATA
-- Reason: HALO-U (Novelty): 'hint02' (score=0.00, src_sp=2.1x, risk=0%) from A_NVMe
SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */ MIN(t.title) AS american_movie FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info AS mi, title AS t WHERE ct.kind  = 'production companies' AND mc.note  not like '%(TV)%' and mc.note like '%(USA)%' AND mi.info  IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 1990 AND t.id = mi.movie_id AND t.id = mc.movie_id AND mc.movie_id = mi.movie_id AND ct.id = mc.company_type_id AND it.id = mi.info_type_id;
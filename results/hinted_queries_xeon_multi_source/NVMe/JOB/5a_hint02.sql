-- HALO Optimized
-- Query: 5a
-- Scenario: Xeon_NVMe
-- Reason: HALO-U (Novelty): 'hint02' (score=0.00, src_sp=1.2x, risk=0%) from A_SATA
SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */ MIN(t.title) AS typical_european_movie FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info AS mi, title AS t WHERE ct.kind  = 'production companies' AND mc.note  like '%(theatrical)%' and mc.note like '%(France)%' AND mi.info  IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German') AND t.production_year > 2005 AND t.id = mi.movie_id AND t.id = mc.movie_id AND mc.movie_id = mi.movie_id AND ct.id = mc.company_type_id AND it.id = mi.info_type_id;
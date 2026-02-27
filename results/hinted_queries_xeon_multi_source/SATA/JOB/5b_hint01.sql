-- HALO Optimized
-- Query: 5b
-- Scenario: Xeon_SATA
-- Reason: HALO-U (Novelty): 'hint01' (score=0.00, src_sp=1.2x, risk=0%) from B_NVMe
SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(t.title) AS american_vhs_movie FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info AS mi, title AS t WHERE ct.kind  = 'production companies' AND mc.note  like '%(VHS)%' and mc.note like '%(USA)%' and mc.note like '%(1994)%' AND mi.info  IN ('USA', 'America') AND t.production_year > 2010 AND t.id = mi.movie_id AND t.id = mc.movie_id AND mc.movie_id = mi.movie_id AND ct.id = mc.company_type_id AND it.id = mi.info_type_id;
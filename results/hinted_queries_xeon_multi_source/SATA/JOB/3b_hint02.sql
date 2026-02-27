-- HALO Optimized
-- Query: 3b
-- Scenario: Xeon_SATA
-- Reason: HALO-U (Novelty): 'hint02' (score=0.00, src_sp=2.8x, risk=0%) from B_NVMe
SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */ MIN(t.title) AS movie_title FROM keyword AS k, movie_info AS mi, movie_keyword AS mk, title AS t WHERE k.keyword  like '%sequel%' AND mi.info  IN ('Bulgaria') AND t.production_year > 2010 AND t.id = mi.movie_id AND t.id = mk.movie_id AND mk.movie_id = mi.movie_id AND k.id = mk.keyword_id;
-- HALO Optimized
-- Query: 4b
-- Scenario: Xeon_NVMe
-- Reason: HALO-U (Novelty): 'hint02' (score=0.00, src_sp=2.3x, risk=0%) from A_NVMe
SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */ MIN(mi_idx.info) AS rating, MIN(t.title) AS movie_title FROM info_type AS it, keyword AS k, movie_info_idx AS mi_idx, movie_keyword AS mk, title AS t WHERE it.info ='rating' AND k.keyword  like '%sequel%' AND mi_idx.info  > '9.0' AND t.production_year > 2010 AND t.id = mi_idx.movie_id AND t.id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND k.id = mk.keyword_id AND it.id = mi_idx.info_type_id;
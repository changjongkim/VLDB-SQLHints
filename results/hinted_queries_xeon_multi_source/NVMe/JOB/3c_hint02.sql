-- HALO Optimized
-- Query: 3c
-- Scenario: Xeon_NVMe
-- Reason: HALO-U (Novelty): 'hint02' (score=0.00, src_sp=5.5x, risk=0%) from A_NVMe
SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */ MIN(t.title) AS movie_title FROM keyword AS k, movie_info AS mi, movie_keyword AS mk, title AS t WHERE k.keyword  like '%sequel%' AND mi.info  IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 1990 AND t.id = mi.movie_id AND t.id = mk.movie_id AND mk.movie_id = mi.movie_id AND k.id = mk.keyword_id;
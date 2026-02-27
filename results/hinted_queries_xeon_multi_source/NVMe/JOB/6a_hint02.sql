-- HALO Optimized
-- Query: 6a
-- Scenario: Xeon_NVMe
-- Reason: HALO-U (Novelty): 'hint02' (score=0.00, src_sp=2.6x, risk=0%) from A_NVMe
SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */ MIN(k.keyword) AS movie_keyword, MIN(n.name) AS actor_name, MIN(t.title) AS marvel_movie FROM cast_info AS ci, keyword AS k, movie_keyword AS mk, name AS n, title AS t WHERE k.keyword = 'marvel-cinematic-universe' AND n.name LIKE '%Downey%Robert%' AND t.production_year > 2010 AND k.id = mk.keyword_id AND t.id = mk.movie_id AND t.id = ci.movie_id AND ci.movie_id = mk.movie_id AND n.id = ci.person_id;
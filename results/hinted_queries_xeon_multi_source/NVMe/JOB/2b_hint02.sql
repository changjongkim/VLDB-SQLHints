-- HALO Optimized
-- Query: 2b
-- Scenario: Xeon_NVMe
-- Reason: HALO-U (Novelty): 'hint02' (score=0.00, src_sp=3.1x, risk=0%) from B_NVMe
SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */ MIN(t.title) AS movie_title FROM company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, title AS t WHERE cn.country_code ='[nl]' AND k.keyword ='character-name-in-title' AND cn.id = mc.company_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mc.movie_id = mk.movie_id;
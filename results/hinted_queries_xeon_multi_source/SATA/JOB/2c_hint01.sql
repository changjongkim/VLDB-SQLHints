-- HALO Optimized
-- Query: 2c
-- Scenario: Xeon_SATA
-- Reason: HALO-U (Novelty): 'hint01' (score=0.00, src_sp=1.1x, risk=0%) from B_NVMe
SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(t.title) AS movie_title FROM company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, title AS t WHERE cn.country_code ='[sm]' AND k.keyword ='character-name-in-title' AND cn.id = mc.company_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mc.movie_id = mk.movie_id;
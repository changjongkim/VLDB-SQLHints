-- HALO Recommended SQL
-- Query     : 2a (JOB)
-- Scenario  : A_NVMe → Xeon_NVMe (AMD EPYC Target)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' selected (src=1.19x, risk_ops=18/66, risk_ratio=27%)
-- Hint Str  : SET_VAR(optimizer_switch="block_nested_loop=off")
--

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(t.title) AS movie_title FROM company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, title AS t WHERE cn.country_code ='[de]' AND k.keyword ='character-name-in-title' AND cn.id = mc.company_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mc.movie_id = mk.movie_id;;

-- HALO Recommended SQL (JOB)
-- Query     : 2a
-- Target    : Intel Xeon Silver 4310 (SATA)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' (src=1.32x, risk=3/11, ratio=27%)
-- Src spdup : 1.32x (best available hint)
--

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(t.title) AS movie_title FROM company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, title AS t WHERE cn.country_code ='[de]' AND k.keyword ='character-name-in-title' AND cn.id = mc.company_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mc.movie_id = mk.movie_id;;

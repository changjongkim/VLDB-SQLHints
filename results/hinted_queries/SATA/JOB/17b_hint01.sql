-- HALO Recommended SQL (JOB)
-- Query     : 17b
-- Target    : Intel Xeon Silver 4310 (SATA)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' CAUTIOUS (src=1.03x, ratio=47%)
-- Src spdup : 1.04x (best available hint)
--

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(n.name) AS member_in_charnamed_movie, MIN(n.name) AS a1 FROM cast_info AS ci, company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, name AS n, title AS t WHERE k.keyword ='character-name-in-title' AND n.name  LIKE 'Z%' AND n.id = ci.person_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND ci.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mc.movie_id = mk.movie_id;;

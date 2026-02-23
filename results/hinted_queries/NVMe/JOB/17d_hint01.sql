-- HALO Recommended SQL (JOB)
-- Query     : 17d
-- Target    : Intel Xeon Silver 4310 (NVMe)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' CAUTIOUS (src=1.05x, ratio=40%)
-- Src spdup : 1.05x (best available hint)
--

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(n.name) AS member_in_charnamed_movie FROM cast_info AS ci, company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, name AS n, title AS t WHERE k.keyword ='character-name-in-title' AND n.name  LIKE '%Bert%' AND n.id = ci.person_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND ci.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mc.movie_id = mk.movie_id;;

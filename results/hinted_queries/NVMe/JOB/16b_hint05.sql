-- HALO Recommended SQL (JOB)
-- Query     : 16b
-- Target    : Intel Xeon Silver 4310 (NVMe)
-- Hint      : hint05
-- Reason    : HALO-R: 'hint05' (src=1.21x, risk=5/17, ratio=29%)
-- Src spdup : 1.21x (best available hint)
--

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY) */ MIN(an.name) AS cool_actor_pseudonym, MIN(t.title) AS series_named_after_char FROM aka_name AS an, cast_info AS ci, company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, name AS n, title AS t WHERE cn.country_code ='[us]' AND k.keyword ='character-name-in-title' AND an.person_id = n.id AND n.id = ci.person_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND an.person_id = ci.person_id AND ci.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mc.movie_id = mk.movie_id;;

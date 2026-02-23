-- HALO Recommended SQL (JOB)
-- Query     : 10c
-- Target    : Intel Xeon Silver 4310 (NVMe)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' CAUTIOUS (src=1.11x, ratio=38%)
-- Src spdup : 1.11x (best available hint)
--

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(chn.name) AS ch, MIN(t.title) AS movie_with_american_producer FROM char_name AS chn, cast_info AS ci, company_name AS cn, company_type AS ct, movie_companies AS mc, role_type AS rt, title AS t WHERE ci.note  like '%(producer)%' AND cn.country_code  = '[us]' AND t.production_year > 1990 AND t.id = mc.movie_id AND t.id = ci.movie_id AND ci.movie_id = mc.movie_id AND chn.id = ci.person_role_id AND rt.id = ci.role_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id;;

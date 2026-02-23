-- HALO Recommended SQL (JOB)
-- Query     : 10a
-- Target    : Intel Xeon Silver 4310 (SATA)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' CAUTIOUS (src=1.03x, ratio=47%)
-- Src spdup : 1.06x (best available hint)
--

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(chn.name) AS uncredited_voiced_character, MIN(t.title) AS russian_movie FROM char_name AS chn, cast_info AS ci, company_name AS cn, company_type AS ct, movie_companies AS mc, role_type AS rt, title AS t WHERE ci.note  like '%(voice)%' and ci.note like '%(uncredited)%' AND cn.country_code  = '[ru]' AND rt.role  = 'actor' AND t.production_year > 2005 AND t.id = mc.movie_id AND t.id = ci.movie_id AND ci.movie_id = mc.movie_id AND chn.id = ci.person_role_id AND rt.id = ci.role_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id;;

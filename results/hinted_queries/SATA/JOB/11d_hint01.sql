-- HALO Recommended SQL (JOB)
-- Query     : 11d
-- Target    : Intel Xeon Silver 4310 (SATA)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' CAUTIOUS (src=1.04x, ratio=50%)
-- Src spdup : 1.04x (best available hint)
--

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(cn.name) AS from_company, MIN(mc.note) AS production_note, MIN(t.title) AS movie_based_on_book FROM company_name AS cn, company_type AS ct, keyword AS k, link_type AS lt, movie_companies AS mc, movie_keyword AS mk, movie_link AS ml, title AS t WHERE cn.country_code  !='[pl]' AND ct.kind  != 'production companies' and ct.kind is not NULL AND k.keyword  in ('sequel', 'revenge', 'based-on-novel') AND mc.note  is not NULL AND t.production_year  > 1950 AND lt.id = ml.link_type_id AND ml.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND ml.movie_id = mk.movie_id AND ml.movie_id = mc.movie_id AND mk.movie_id = mc.movie_id;;

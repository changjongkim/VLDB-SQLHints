-- HALO Recommended SQL (JOB)
-- Query     : 5b
-- Target    : Intel Xeon Silver 4310 (SATA)
-- Hint      : NATIVE
-- Reason    : HALO-R: all too risky → NATIVE [hint01(75%), hint02(75%), hint05(75%)]
-- Src spdup : 1.23x (best available hint)
--

SELECT MIN(t.title) AS american_vhs_movie FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info AS mi, title AS t WHERE ct.kind  = 'production companies' AND mc.note  like '%(VHS)%' and mc.note like '%(USA)%' and mc.note like '%(1994)%' AND mi.info  IN ('USA', 'America') AND t.production_year > 2010 AND t.id = mi.movie_id AND t.id = mc.movie_id AND mc.movie_id = mi.movie_id AND ct.id = mc.company_type_id AND it.id = mi.info_type_id;;

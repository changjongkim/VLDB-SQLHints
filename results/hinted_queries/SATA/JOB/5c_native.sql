-- HALO Recommended SQL (JOB)
-- Query     : 5c
-- Target    : Intel Xeon Silver 4310 (SATA)
-- Hint      : NATIVE
-- Reason    : HALO-R: all too risky → NATIVE [hint01(54%), hint02(54%), hint05(54%)]
-- Src spdup : 1.14x (best available hint)
--

SELECT MIN(t.title) AS american_movie FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info AS mi, title AS t WHERE ct.kind  = 'production companies' AND mc.note  not like '%(TV)%' and mc.note like '%(USA)%' AND mi.info  IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 1990 AND t.id = mi.movie_id AND t.id = mc.movie_id AND mc.movie_id = mi.movie_id AND ct.id = mc.company_type_id AND it.id = mi.info_type_id;;

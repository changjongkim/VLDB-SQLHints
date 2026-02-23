-- HALO Recommended SQL (JOB)
-- Query     : 12b
-- Target    : Intel Xeon Silver 4310 (SATA)
-- Hint      : NATIVE
-- Reason    : HALO-R: all too risky → NATIVE [hint01(76%), hint02(76%), hint05(76%)]
-- Src spdup : 1.01x (best available hint)
--

SELECT MIN(mi.info) AS budget, MIN(t.title) AS unsuccsessful_movie FROM company_name AS cn, company_type AS ct, info_type AS it1, info_type AS it2, movie_companies AS mc, movie_info AS mi, movie_info_idx AS mi_idx, title AS t WHERE cn.country_code ='[us]' AND ct.kind  is not NULL and (ct.kind ='production companies' or ct.kind = 'distributors') AND it1.info ='budget' AND it2.info ='bottom 10 rank' AND t.production_year >2000 AND (t.title LIKE 'Birdemic%' OR t.title LIKE '%Movie%') AND t.id = mi.movie_id AND t.id = mi_idx.movie_id AND mi.info_type_id = it1.id AND mi_idx.info_type_id = it2.id AND t.id = mc.movie_id AND ct.id = mc.company_type_id AND cn.id = mc.company_id AND mc.movie_id = mi.movie_id AND mc.movie_id = mi_idx.movie_id AND mi.movie_id = mi_idx.movie_id;;

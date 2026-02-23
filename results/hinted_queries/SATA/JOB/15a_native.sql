-- HALO Recommended SQL (JOB)
-- Query     : 15a
-- Target    : Intel Xeon Silver 4310 (SATA)
-- Hint      : NATIVE
-- Reason    : HALO-R: all too risky → NATIVE [hint01(68%), hint02(68%), hint05(68%)]
-- Src spdup : 1.08x (best available hint)
--

SELECT MIN(mi.info) AS release_date, MIN(t.title) AS internet_movie FROM aka_title AS at, company_name AS cn, company_type AS ct, info_type AS it1, keyword AS k, movie_companies AS mc, movie_info AS mi, movie_keyword AS mk, title AS t WHERE cn.country_code  = '[us]' AND it1.info  = 'release dates' AND mc.note  like '%(200%)%' and mc.note like '%(worldwide)%' AND mi.note  like '%internet%' AND mi.info  like 'USA:% 200%' AND t.production_year  > 2000 AND t.id = at.movie_id AND t.id = mi.movie_id AND t.id = mk.movie_id AND t.id = mc.movie_id AND mk.movie_id = mi.movie_id AND mk.movie_id = mc.movie_id AND mk.movie_id = at.movie_id AND mi.movie_id = mc.movie_id AND mi.movie_id = at.movie_id AND mc.movie_id = at.movie_id AND k.id = mk.keyword_id AND it1.id = mi.info_type_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id;;

-- HALO Recommended SQL (JOB)
-- Query     : 2d
-- Target    : Intel Xeon Silver 4310 (NVMe)
-- Hint      : NATIVE
-- Reason    : HALO-R: all too risky → NATIVE [hint01(82%), hint02(82%), hint05(82%)]
-- Src spdup : 1.00x (best available hint)
--

SELECT MIN(t.title) AS movie_title FROM company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, title AS t WHERE cn.country_code ='[us]' AND k.keyword ='character-name-in-title' AND cn.id = mc.company_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mc.movie_id = mk.movie_id;;

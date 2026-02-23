-- HALO Recommended SQL (JOB)
-- Query     : 6a
-- Target    : Intel Xeon Silver 4310 (NVMe)
-- Hint      : NATIVE
-- Reason    : HALO-R: all too risky → NATIVE [hint01(75%), hint02(75%), hint05(75%)]
-- Src spdup : 0.94x (best available hint)
--

SELECT MIN(k.keyword) AS movie_keyword, MIN(n.name) AS actor_name, MIN(t.title) AS marvel_movie FROM cast_info AS ci, keyword AS k, movie_keyword AS mk, name AS n, title AS t WHERE k.keyword = 'marvel-cinematic-universe' AND n.name LIKE '%Downey%Robert%' AND t.production_year > 2010 AND k.id = mk.keyword_id AND t.id = mk.movie_id AND t.id = ci.movie_id AND ci.movie_id = mk.movie_id AND n.id = ci.person_id;;

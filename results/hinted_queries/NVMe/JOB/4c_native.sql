-- HALO Recommended SQL (JOB)
-- Query     : 4c
-- Target    : Intel Xeon Silver 4310 (NVMe)
-- Hint      : NATIVE
-- Reason    : HALO-R: all too risky → NATIVE [hint01(77%), hint02(77%), hint05(77%)]
-- Src spdup : 1.20x (best available hint)
--

SELECT MIN(mi_idx.info) AS rating, MIN(t.title) AS movie_title FROM info_type AS it, keyword AS k, movie_info_idx AS mi_idx, movie_keyword AS mk, title AS t WHERE it.info ='rating' AND k.keyword  like '%sequel%' AND mi_idx.info  > '2.0' AND t.production_year > 1990 AND t.id = mi_idx.movie_id AND t.id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND k.id = mk.keyword_id AND it.id = mi_idx.info_type_id;;

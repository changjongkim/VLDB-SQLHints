-- HALO Recommended SQL (JOB)
-- Query     : 18a
-- Target    : Intel Xeon Silver 4310 (SATA)
-- Hint      : NATIVE
-- Reason    : HALO-R: all too risky → NATIVE [hint01(80%), hint02(80%), hint05(80%)]
-- Src spdup : 1.00x (best available hint)
--

SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(t.title) AS movie_title FROM cast_info AS ci, info_type AS it1, info_type AS it2, movie_info AS mi, movie_info_idx AS mi_idx, name AS n, title AS t WHERE ci.note  in ('(producer)', '(executive producer)') AND it1.info  = 'budget' AND it2.info  = 'votes' AND n.gender  = 'm' and n.name like '%Tim%' AND t.id = mi.movie_id AND t.id = mi_idx.movie_id AND t.id = ci.movie_id AND ci.movie_id = mi.movie_id AND ci.movie_id = mi_idx.movie_id AND mi.movie_id = mi_idx.movie_id AND n.id = ci.person_id AND it1.id = mi.info_type_id AND it2.id = mi_idx.info_type_id;;

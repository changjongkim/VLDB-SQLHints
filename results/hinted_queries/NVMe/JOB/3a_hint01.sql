-- HALO Recommended SQL (JOB)
-- Query     : 3a
-- Target    : Intel Xeon Silver 4310 (NVMe)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' CAUTIOUS (src=1.15x, ratio=50%)
-- Src spdup : 1.15x (best available hint)
--

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(t.title) AS movie_title FROM keyword AS k, movie_info AS mi, movie_keyword AS mk, title AS t WHERE k.keyword  like '%sequel%' AND mi.info  IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German') AND t.production_year > 2005 AND t.id = mi.movie_id AND t.id = mk.movie_id AND mk.movie_id = mi.movie_id AND k.id = mk.keyword_id;;

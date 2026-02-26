-- HALO Recommended SQL
-- Query     : 6e (JOB)
-- Target HW : Xeon_NVMe
-- Mode      : Multi-Source Global Selection
-- Hint      : hint01
-- From Src  : B_NVMe
-- Reason    : HALO-R (Multi-Source): 'hint01' selected from B_NVMe (src_speedup=31.16x, risk=17%)
-- Generated : 2026-02-26
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(k.keyword) AS movie_keyword, MIN(n.name) AS actor_name, MIN(t.title) AS marvel_movie FROM cast_info AS ci, keyword AS k, movie_keyword AS mk, name AS n, title AS t WHERE k.keyword = 'marvel-cinematic-universe' AND n.name LIKE '%Downey%Robert%' AND t.production_year > 2000 AND k.id = mk.keyword_id AND t.id = mk.movie_id AND t.id = ci.movie_id AND ci.movie_id = mk.movie_id AND n.id = ci.person_id;;

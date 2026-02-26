-- HALO Recommended SQL
-- Query     : 20b (JOB)
-- Target HW : Xeon_NVMe
-- Mode      : Multi-Source Global Selection
-- Hint      : hint05
-- From Src  : B_SATA
-- Reason    : HALO-R (Multi-Source): 'hint05' selected from B_SATA (src_speedup=1.10x, risk=17%)
-- Generated : 2026-02-26
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY) */ MIN(t.title) AS complete_downey_ironman_movie FROM complete_cast AS cc, comp_cast_type AS cct1, comp_cast_type AS cct2, char_name AS chn, cast_info AS ci, keyword AS k, kind_type AS kt, movie_keyword AS mk, name AS n, title AS t WHERE cct1.kind  = 'cast' AND cct2.kind  like '%complete%' AND chn.name  not like '%Sherlock%' and (chn.name like '%Tony%Stark%' or chn.name like '%Iron%Man%') AND k.keyword  in ('superhero', 'sequel', 'second-part', 'marvel-comics', 'based-on-comic', 'tv-special', 'fight', 'violence') AND kt.kind  = 'movie' AND n.name  LIKE '%Downey%Robert%' AND t.production_year  > 2000 AND kt.id = t.kind_id AND t.id = mk.movie_id AND t.id = ci.movie_id AND t.id = cc.movie_id AND mk.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND ci.movie_id = cc.movie_id AND chn.id = ci.person_role_id AND n.id = ci.person_id AND k.id = mk.keyword_id AND cct1.id = cc.subject_id AND cct2.id = cc.status_id;;

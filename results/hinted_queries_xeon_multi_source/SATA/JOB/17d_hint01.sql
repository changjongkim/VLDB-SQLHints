-- HALO Recommended SQL
-- Query     : 17d (JOB)
-- Target HW : Xeon_SATA
-- Mode      : Multi-Source Global Selection
-- Hint      : hint01
-- From Src  : B_NVMe
-- Reason    : HALO-R (Multi-Source): 'hint01' selected from B_NVMe (src_speedup=1.89x, risk=13%)
-- Generated : 2026-02-26
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(n.name) AS member_in_charnamed_movie FROM cast_info AS ci, company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, name AS n, title AS t WHERE k.keyword ='character-name-in-title' AND n.name  LIKE '%Bert%' AND n.id = ci.person_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND ci.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mc.movie_id = mk.movie_id;;

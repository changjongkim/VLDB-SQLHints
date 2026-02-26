-- HALO Recommended SQL
-- Query     : 17a (JOB)
-- Target HW : Xeon_NVMe
-- Mode      : Multi-Source Global Selection
-- Hint      : hint02
-- From Src  : A_SATA
-- Reason    : HALO-R (Multi-Source): 'hint02' selected from A_SATA (src_speedup=4.45x, risk=28%)
-- Generated : 2026-02-26
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */ MIN(n.name) AS member_in_charnamed_american_movie, MIN(n.name) AS a1 FROM cast_info AS ci, company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, name AS n, title AS t WHERE cn.country_code ='[us]' AND k.keyword ='character-name-in-title' AND n.name  LIKE 'B%' AND n.id = ci.person_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND ci.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mc.movie_id = mk.movie_id;;

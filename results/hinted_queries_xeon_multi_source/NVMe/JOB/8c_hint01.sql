-- HALO Recommended SQL
-- Query     : 8c (JOB)
-- Target HW : Xeon_NVMe
-- Mode      : Multi-Source Global Selection
-- Hint      : hint01
-- From Src  : A_SATA
-- Reason    : HALO-R (Multi-Source): 'hint01' selected from A_SATA (src_speedup=1.68x, risk=29%)
-- Generated : 2026-02-26
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(a1.name) AS writer_pseudo_name, MIN(t.title) AS movie_title FROM aka_name AS a1, cast_info AS ci, company_name AS cn, movie_companies AS mc, name AS n1, role_type AS rt, title AS t WHERE cn.country_code ='[us]' AND rt.role ='writer' AND a1.person_id = n1.id AND n1.id = ci.person_id AND ci.movie_id = t.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND ci.role_id = rt.id AND a1.person_id = ci.person_id AND ci.movie_id = mc.movie_id;;

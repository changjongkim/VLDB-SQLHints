-- HALO Recommended SQL
-- Query     : 5c (JOB)
-- Target HW : Xeon_SATA
-- Mode      : Multi-Source Global Selection
-- Hint      : hint01
-- From Src  : B_NVMe
-- Reason    : HALO-R (Multi-Source): 'hint01' selected from B_NVMe (src_speedup=1.36x, risk=19%)
-- Generated : 2026-02-26
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(t.title) AS american_movie FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info AS mi, title AS t WHERE ct.kind  = 'production companies' AND mc.note  not like '%(TV)%' and mc.note like '%(USA)%' AND mi.info  IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 1990 AND t.id = mi.movie_id AND t.id = mc.movie_id AND mc.movie_id = mi.movie_id AND ct.id = mc.company_type_id AND it.id = mi.info_type_id;;

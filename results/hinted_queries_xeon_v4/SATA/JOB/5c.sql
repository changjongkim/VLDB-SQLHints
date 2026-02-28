-- HALO v4 Recommended SQL (Performance-Focused)
-- Query     : 5c
-- Scenario  : Xeon_SATA
-- Mode      : HALO-P v4 (Power/Performance Mode)
-- Hint      : hint01
-- Risk Level : ORANGE
-- Reason    : Performance candidate selected (Gain=2.54)
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(t.title) AS american_movie FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info AS mi, title AS t WHERE ct.kind  = 'production companies' AND mc.note  not like '%(TV)%' and mc.note like '%(USA)%' AND mi.info  IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 1990 AND t.id = mi.movie_id AND t.id = mc.movie_id AND mc.movie_id = mi.movie_id AND ct.id = mc.company_type_id AND it.id = mi.info_type_id;

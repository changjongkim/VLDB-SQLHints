-- HALO v4 Recommended SQL (Performance-Focused)
-- Query     : 3c
-- Scenario  : Xeon_NVMe
-- Mode      : HALO-P v4 (Power/Performance Mode)
-- Hint      : hint01
-- Risk Level : ORANGE
-- Reason    : Performance candidate selected (Gain=7.63)
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(t.title) AS movie_title FROM keyword AS k, movie_info AS mi, movie_keyword AS mk, title AS t WHERE k.keyword  like '%sequel%' AND mi.info  IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 1990 AND t.id = mi.movie_id AND t.id = mk.movie_id AND mk.movie_id = mi.movie_id AND k.id = mk.keyword_id;

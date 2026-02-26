-- HALO Recommended SQL
-- Query     : 33a (JOB)
-- Target HW : Xeon_SATA
-- Mode      : Multi-Source Global Selection
-- Hint      : hint05
-- From Src  : B_SATA
-- Reason    : HALO-R (Multi-Source): 'hint05' selected from B_SATA (src_speedup=1.43x, risk=25%)
-- Generated : 2026-02-26
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") NO_RANGE_OPTIMIZATION(t1 PRIMARY) */ MIN(cn1.name) AS first_company, MIN(cn2.name) AS second_company, MIN(mi_idx1.info) AS first_rating, MIN(mi_idx2.info) AS second_rating, MIN(t1.title) AS first_movie, MIN(t2.title) AS second_movie FROM company_name AS cn1, company_name AS cn2, info_type AS it1, info_type AS it2, kind_type AS kt1, kind_type AS kt2, link_type AS lt, movie_companies AS mc1, movie_companies AS mc2, movie_info_idx AS mi_idx1, movie_info_idx AS mi_idx2, movie_link AS ml, title AS t1, title AS t2 WHERE cn1.country_code  = '[us]' AND it1.info  = 'rating' AND it2.info  = 'rating' AND kt1.kind  in ('tv series') AND kt2.kind  in ('tv series') AND lt.link  in ('sequel', 'follows', 'followed by') AND mi_idx2.info  < '3.0' AND t2.production_year  between 2005 and 2008 AND lt.id = ml.link_type_id AND t1.id = ml.movie_id AND t2.id = ml.linked_movie_id AND it1.id = mi_idx1.info_type_id AND t1.id = mi_idx1.movie_id AND kt1.id = t1.kind_id AND cn1.id = mc1.company_id AND t1.id = mc1.movie_id AND ml.movie_id = mi_idx1.movie_id AND ml.movie_id = mc1.movie_id AND mi_idx1.movie_id = mc1.movie_id AND it2.id = mi_idx2.info_type_id AND t2.id = mi_idx2.movie_id AND kt2.id = t2.kind_id AND cn2.id = mc2.company_id AND t2.id = mc2.movie_id AND ml.linked_movie_id = mi_idx2.movie_id AND ml.linked_movie_id = mc2.movie_id AND mi_idx2.movie_id = mc2.movie_id;;

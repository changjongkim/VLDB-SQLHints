-- HALO Recommended SQL
-- Query     : 13a (JOB)
-- Scenario  : A_NVMe → B_SATA (AMD EPYC Target)
-- Hint      : hint01
-- Reason    : HALO-R: 'hint01' selected cautiously (src=2.59x, risk_ratio=43% HIGH)
-- Hint Str  : SET_VAR(optimizer_switch="block_nested_loop=off")
--

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off") */ MIN(mi.info) AS release_date, MIN(miidx.info) AS rating, MIN(t.title) AS german_movie FROM company_name AS cn, company_type AS ct, info_type AS it, info_type AS it2, kind_type AS kt, movie_companies AS mc, movie_info AS mi, movie_info_idx AS miidx, title AS t WHERE cn.country_code ='[de]' AND ct.kind ='production companies' AND it.info ='rating' AND it2.info ='release dates' AND kt.kind ='movie' AND mi.movie_id = t.id AND it2.id = mi.info_type_id AND kt.id = t.kind_id AND mc.movie_id = t.id AND cn.id = mc.company_id AND ct.id = mc.company_type_id AND miidx.movie_id = t.id AND it.id = miidx.info_type_id AND mi.movie_id = miidx.movie_id AND mi.movie_id = mc.movie_id AND miidx.movie_id = mc.movie_id;;

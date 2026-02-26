-- HALO Recommended SQL
-- Query     : 7b (JOB)
-- Scenario  : A_NVMe → Xeon_NVMe (AMD EPYC Target)
-- Hint      : NATIVE (no hint)
-- Reason    : HALO-R: All hints too risky. hint01(69%), hint02(67%) → NATIVE
-- Hint Str  : N/A
--

SELECT MIN(n.name) AS of_person, MIN(t.title) AS biography_movie FROM aka_name AS an, cast_info AS ci, info_type AS it, link_type AS lt, movie_link AS ml, name AS n, person_info AS pi, title AS t WHERE an.name LIKE '%a%' AND it.info ='mini biography' AND lt.link ='features' AND n.name_pcode_cf LIKE 'D%' AND n.gender='m' AND pi.note ='Volker Boehm' AND t.production_year BETWEEN 1980 AND 1984 AND n.id = an.person_id AND n.id = pi.person_id AND ci.person_id = n.id AND t.id = ci.movie_id AND ml.linked_movie_id = t.id AND lt.id = ml.link_type_id AND it.id = pi.info_type_id AND pi.person_id = an.person_id AND pi.person_id = ci.person_id AND an.person_id = ci.person_id AND ci.movie_id = ml.linked_movie_id;;

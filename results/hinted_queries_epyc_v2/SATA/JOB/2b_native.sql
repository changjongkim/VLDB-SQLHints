-- HALO Recommended SQL
-- Query     : 2b (JOB)
-- Scenario  : A_NVMe → B_SATA (AMD EPYC Target)
-- Hint      : NATIVE (no hint)
-- Reason    : HALO-R: All hints too risky. hint01(55%), hint02(45%) → NATIVE
-- Hint Str  : N/A
--

SELECT MIN(t.title) AS movie_title FROM company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, title AS t WHERE cn.country_code ='[nl]' AND k.keyword ='character-name-in-title' AND cn.id = mc.company_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mc.movie_id = mk.movie_id;;

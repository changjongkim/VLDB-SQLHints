-- HALO Recommended SQL
-- Query     : 19a (JOB)
-- Target HW : Xeon_NVMe
-- Mode      : Multi-Source Global Selection
-- Hint      : hint02
-- From Src  : A_NVMe
-- Reason    : HALO-U (Novelty): 'hint02' selected (score=0.00, src_sp=2.14x, risk=10%)
-- Generated : 2026-02-26
======================================================================

SELECT /*+ SET_VAR(optimizer_switch="block_nested_loop=off,batched_key_access=on") SET_VAR(optimizer_switch="mrr=on,mrr_cost_based=off") */ MIN(n.name) AS voicing_actress, MIN(t.title) AS voiced_movie FROM aka_name AS an, char_name AS chn, cast_info AS ci, company_name AS cn, info_type AS it, movie_companies AS mc, movie_info AS mi, name AS n, role_type AS rt, title AS t WHERE ci.note  in ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code ='[us]' AND it.info  = 'release dates' AND mc.note  is not NULL and (mc.note like '%(USA)%' or mc.note like '%(worldwide)%') AND mi.info  is not null and (mi.info like 'Japan:%200%' or mi.info like 'USA:%200%') AND n.gender ='f' and n.name like '%Ang%' AND rt.role ='actress' AND t.production_year  between 2005 and 2009 AND t.id = mi.movie_id AND t.id = mc.movie_id AND t.id = ci.movie_id AND mc.movie_id = ci.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND cn.id = mc.company_id AND it.id = mi.info_type_id AND n.id = ci.person_id AND rt.id = ci.role_id AND n.id = an.person_id AND ci.person_id = an.person_id AND chn.id = ci.person_role_id;;

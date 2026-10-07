import json
import unittest

from finrate_completed100_v1 import select_metadata, grade_one, summarize


class FakeJudge:
    def __init__(self, points, label='PARTIAL'):
        self.points=points;self.label=label;self.calls=0;self.schema=None;self.prompt=None

    def invoke(self,prompt,folder,*,schema):
        self.calls+=1;self.schema=schema;self.prompt=prompt
        return json.dumps({'label':self.label,'key_points_met':self.points,'citation_support':'no_citations',
                           'temporal_error':False,'entity_error':False,'reason':'fixture'})


class Completed100Test(unittest.TestCase):
    def test_selection_is_fixed_balanced_and_independent_of_text(self):
        pools={t:[{'q_id':t+str(i),'task':t,'question':'old','answer':'old','key_points':['old']} for i in range(80)] for t in ['LT-QA','EC-QA']}
        selected=select_metadata(pools)
        self.assertEqual(len(selected),100)
        self.assertEqual(sum(q['task']=='LT-QA' for q in selected),50)
        changed={t:[{**q,'question':'changed','answer':'changed','key_points':['changed']*5} for q in reversed(v)] for t,v in pools.items()}
        self.assertEqual([q['q_id'] for q in selected],[q['q_id'] for q in select_metadata(changed)])

    def test_insufficient_pool_rejected_without_replacement(self):
        pools={t:[{'q_id':t+str(i),'task':t} for i in range(49)] for t in ['LT-QA','EC-QA']}
        with self.assertRaises(ValueError):select_metadata(pools)

    def test_grading_schema_has_exact_count_and_method_name_is_absent(self):
        q={'question':'Question?','answer':'Reference','key_points':['a','b','c']}
        pred={'status':'completed','answer':{'answer':'Candidate','claims':[]},'evidence':[],'condition':'SECRET_METHOD_NAME'}
        model=FakeJudge([True,False,True]);row=grade_one(q,pred,model,None)
        self.assertAlmostEqual(row['key_point_fraction'],2/3)
        self.assertFalse(row['correct'])
        self.assertEqual(model.schema['properties']['key_points_met']['minItems'],3)
        self.assertEqual(model.schema['properties']['key_points_met']['maxItems'],3)
        self.assertNotIn('SECRET_METHOD_NAME',model.prompt)

    def test_generation_failure_is_zero_without_judge_call(self):
        model=FakeJudge([])
        row=grade_one({'key_points':['a','b']},{'status':'failed','error':'provider unavailable'},model,None)
        self.assertEqual(model.calls,0);self.assertEqual(row['key_point_fraction'],0)

    def test_invalid_judge_output_stays_unassessed(self):
        q={'question':'Q','answer':'A','key_points':['a','b']}
        pred={'status':'completed','answer':{'answer':'A','claims':[]},'evidence':[]}
        row=grade_one(q,pred,FakeJudge([True]),None)
        self.assertIsNone(row['correct']);self.assertIsNone(row['key_point_fraction'])

    def test_summary_keeps_failures_and_unknowns_in_denominator(self):
        rows=[{'correct':True,'key_point_fraction':1,'generation_status':'completed','label':'CORRECT'},
              {'correct':False,'key_point_fraction':0,'generation_status':'failed','label':'FAILURE'},
              {'correct':None,'key_point_fraction':None,'generation_status':'completed','label':'UNASSESSED'}]
        report=summarize(rows)
        self.assertEqual(report['n'],3)
        self.assertAlmostEqual(report['accuracy_lower'],1/3)
        self.assertAlmostEqual(report['accuracy_upper'],2/3)
        self.assertAlmostEqual(report['key_point_lower'],1/3)
        self.assertEqual(report['generation_failures'],1)


if __name__=='__main__':unittest.main()

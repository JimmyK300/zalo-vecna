import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from datamate_bench.data import ReadOnlySQL
from verification.checks import inspect,gate
from verification.reviewer import apply_review,payload


class VerificationTests(unittest.TestCase):
    def test_unanswerable_requires_structured_witness(self):
        result={'prediction':{'type':'unanswerable'},'trace':[]}
        layer={'date_min':'2025-01-01','date_max':'2025-12-31'}
        outside={'question':'Doanh thu tháng 1/2024','as_of_date':'2025-01-01'}
        inside={'question':'Doanh thu tháng 1/2025','as_of_date':'2025-01-01'}
        self.assertEqual(inspect(outside,result,layer),[])
        self.assertIn('unanswerable_not_certified',[i['code'] for i in inspect(inside,result,layer)])

    def test_growth_is_not_a_bounded_share(self):
        prediction={'type':'percent','answer':150}
        result={'prediction':prediction,'trace':[{'raw_response':'{"action":"metric","operation":"revenue_growth"}',
                          'prediction':prediction,'tool_output':{'prediction':prediction}}]}
        q={'question':'Doanh thu tăng bao nhiêu phần trăm?','as_of_date':'2025-01-01'}
        self.assertEqual(inspect(q,result,{}),[])

    def test_schema_row_count_is_not_a_result_row_list(self):
        q={'question':'Tổng tiền?','as_of_date':'2025-01-01'}
        result={'prediction':{'type':'abstain'},'trace':[{'raw_response':'{"action":"schema","table":"t01"}',
                         'tool_output':{'columns':['x'],'rows':500}}]}
        manifest={'tables':{},'sales_layer':{}}
        data=payload(q,result,[],'','intro',manifest)
        self.assertEqual(data['evidence'][0]['output']['rows'],500)

    def test_concentration_is_not_percentage_request(self):
        q={'question':'Giá bán Thuốc A 0,1% ngày 01/01/2025 là bao nhiêu?','as_of_date':'2025-01-01'}
        result={'prediction':{'type':'number','answer':58000},'trace':[]}
        self.assertEqual(inspect(q,result,{}),[])
        q['question']='Doanh thu tăng bao nhiêu phần trăm?'
        issues=inspect(q,result,{})
        self.assertEqual(gate(result['prediction'],issues),{'type':'abstain'})

    def test_code_gate_catches_fractional_money_without_guessing(self):
        issues=inspect({'question':'Trung bình mỗi đơn bao nhiêu tiền?','as_of_date':'2025-01-01'},
                       {'prediction':{'type':'number','answer':123.6},'trace':[]},{})
        self.assertIn('fractional_money',[i['code'] for i in issues])
        self.assertEqual(gate({'type':'number','answer':123.6},issues),{'type':'abstain'})

    def test_review_executes_sql_instead_of_accepting_model_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.sqlite';con=sqlite3.connect(path)
            con.execute('CREATE TABLE amounts(value)');con.executemany('INSERT INTO amounts VALUES (?)',[(30,),(70,)])
            con.commit();con.close();sql=ReadOnlySQL(path)
            q={'question':'Tổng tiền?','as_of_date':'2025-01-01'}
            original={'prediction':{'type':'number','answer':20},'trace':[]}
            try:
                response={'decision':'replace','repair':{'action':'sql','sql':'SELECT SUM(value) FROM amounts','type':'number'}}
                result=apply_review(json.dumps(response),q,original,[],sql,{})
                self.assertEqual(result['prediction'],{'type':'number','answer':100})
                self.assertTrue(result['repair_executed'])
                response['repair']['answer']=999
                result=apply_review(json.dumps(response),q,original,[],sql,{})
                self.assertEqual(result['prediction'],original['prediction'])
                self.assertIn('review_error',result)
                response={'decision':'replace','repair':{'action':'sql','sql':'DELETE FROM amounts','type':'number'}}
                result=apply_review(json.dumps(response),q,original,[],sql,{})
                self.assertEqual(result['prediction'],original['prediction'])
                self.assertEqual(sql.query('SELECT COUNT(*) FROM amounts')['rows'][0][0],2)
            finally:sql.close()

    def test_invalid_review_preserves_only_code_passed_candidate(self):
        q={'question':'Tổng tiền?','as_of_date':'2025-01-01'}
        original={'prediction':{'type':'number','answer':10},'trace':[]}
        self.assertEqual(apply_review('bad',q,original,[],None,{})['prediction'],original['prediction'])
        issues=[{'code':'invalid','severity':'hard','evidence':'test'}]
        self.assertEqual(apply_review('bad',q,original,issues,None,{})['prediction'],{'type':'abstain'})


if __name__=='__main__':unittest.main()

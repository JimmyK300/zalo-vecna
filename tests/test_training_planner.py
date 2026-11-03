import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from local_training.planner import reward,parse_action,messages


class TrainingPlannerTests(unittest.TestCase):
    def test_reward_requires_correct_executed_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'facts.sqlite';con=sqlite3.connect(path)
            con.execute('CREATE TABLE orders(order_date,revenue,valid_sale,status)')
            con.executemany('INSERT INTO orders VALUES (?,?,?,?)',[('2025-01-01',100,1,'completed'),('2025-01-01',900,0,'cancelled')])
            con.commit();con.close()
            business={'db':str(path),'layer':{'date_min':'2025-01-01','date_max':'2025-01-31','can_count_orders':True}}
            question={'question':'Doanh thu tháng 1/2025','as_of_date':'2025-01-31'}
            reference={'type':'number','answer':100}
            self.assertEqual(reward('{"action":"metric","operation":"revenue"}',question,business,reference)[0],1)
            self.assertEqual(reward('{"action":"metric","operation":"count_orders"}',question,business,reference)[0],-1)
            self.assertEqual(reward('{"action":"metric","operation":"revenue","answer":100}',question,business,reference)[0],-1)
            self.assertEqual(reward('invalid json',question,business,reference)[0],-1)
            self.assertEqual(reward('{"action":"abstain"}',question,business,reference)[0],0)

    def test_payload_has_no_reference_or_identifiers(self):
        question={'id':'internal-only','business_id':'internal-shop','question':'hqua dthu bn','as_of_date':'2025-01-02'}
        business={'products':[],'product_groups':[],'customer_groups':[],'channels':[],
                  'payment_methods':[],'capabilities':{'has_sales_lines':True}}
        result=messages(question,business)
        payload=json.loads(result[1]['content'])
        self.assertEqual(payload['question'],question['question'])
        self.assertNotIn('reference',payload);self.assertNotIn('answer',payload)
        self.assertNotIn('internal-only',result[1]['content']);self.assertNotIn('internal-shop',result[1]['content'])

    def test_parser_cannot_inject_sql_or_answer(self):
        for value in ['{"action":"sql","sql":"SELECT 100"}','{"action":"finish","answer":100}',
                      '{"action":"abstain","answer":100}']:
            with self.assertRaises(ValueError):parse_action(value)


if __name__=='__main__':unittest.main()

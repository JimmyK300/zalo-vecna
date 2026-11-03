"""Conservative checks, independent of reference answers and model confidence."""
import json
import math
import re

from datamate_bench.data import fold
from datamate_bench.scoring import validate
from datamate_bench.temporal import resolve
from datamate_bench.metrics import SUPPORTED


def parsed(entry):
    try:
        action=json.loads(entry.get('raw_response','{}'))
        if not isinstance(action,dict):return {}
        if action.get('action') in SUPPORTED:
            return {**action,'operation':action['action'],'action':'metric'}
        return action
    except ValueError:return {}


def metric_evidence(result):
    for entry in reversed(result.get('trace',[])):
        if 'prediction' in entry and parsed(entry).get('action')=='metric':
            return parsed(entry),entry.get('tool_output',{})
    return {},{}


def inspect(question,result,layer):
    issues=[];prediction=result.get('prediction',{})
    def add(code,evidence,severity='hard'):
        issues.append({'code':code,'severity':severity,'evidence':evidence})
    if not validate(prediction):
        add('invalid_typed_answer','Prediction does not satisfy the typed output contract')
        return issues
    kind=prediction['type'];q=fold(question['question']);action,output=metric_evidence(result)
    op=action.get('operation')
    if kind=='abstain':
        add('no_answer','No answer to verify; reviewer may attempt one supported computation','warning')
        return issues
    if kind=='unanswerable':
        period=resolve(question['question'],question['as_of_date'])
        purchase=op in {'purchase_total','purchase_average'}
        a=layer.get('purchase_date_min' if purchase else 'date_min')
        b=layer.get('purchase_date_max' if purchase else 'date_max')
        outside=bool(period and a and b and (period[1]<a or period[0]>b))
        missing_purchase=purchase and not layer.get('has_purchase_data')
        # A zero-row query alone cannot prove absent information. This deliberately
        # abstains on claims that the checker cannot certify, including some true ones.
        if not (outside or missing_purchase):
            add('unanswerable_not_certified','No structured witness of a wholly outside period or absent purchase source')
        return issues
    percent_text=re.sub(r'\d+(?:[.,]\d+)?\s*%','',q)
    asks_percent=bool(re.search(r'%|phan tram|\bti le\b|\bty le\b',percent_text))
    if asks_percent and kind!='percent':
        add('percent_type_mismatch','Question explicitly asks a percentage')
    if re.search(r'doanh thu|dthu',q) and re.search(r'lien tuc|luon cao hon',q) and kind!='boolean':
        add('trend_type_mismatch','Question asks whether revenue increases continuously')
    multiple_revenues=re.search(r'(?:doanh thu|dthu).*la bao nhieu.*(?:doanh thu|dthu).*la bao nhieu',q)
    if multiple_revenues and kind!='list':
        add('multiple_values_type','Question explicitly requests multiple revenue values in order')
    if kind=='number' and re.search(r'\btien\b|doanh thu|dthu|gia nhap|gia ban|don gia',q) and not asks_percent:
        value=prediction['answer']
        if math.isfinite(value) and abs(value-round(value))>1e-7:
            add('fractional_money','A monetary answer must be rounded to whole VND')
    if kind=='percent' and op in {'cancel_rate','product_group_share','customer_group_share','prescription_share','revenue_share'}:
        if not 0<=prediction['answer']<=100:
            add('share_outside_range','A share/cancellation percentage must lie between 0 and 100')
    if op:
        if re.search(r'gia nhap|\bnhap\b.*binh quan',q) and op=='sale_price':
            add('purchase_vs_sale_price','Purchase price requested but sale-price operation used')
        if re.search(r'gia ban|don gia ban',q) and op=='purchase_average':
            add('sale_vs_purchase_price','Sale price requested but purchase-price operation used')
        if re.search(r'\bca sang\b|\bca chieu\b',q) and op in {'revenue','count_orders','average_order'}:
            add('missing_shift_filter','This metric has no time-of-day parameter')
        if 'hoat chat' in q and op in {'revenue','product_revenue','prescription_revenue','product_group_revenue'}:
            add('missing_ingredient_filter','The selected metric does not filter active ingredient')
        if re.search(r'khach.*(?:nhieu don|nhieu.*don nhat)',q) and op=='top_customers':
            add('orders_vs_spend_ranking','top_customers ranks spending, not order count')
        if output.get('prediction')!=prediction:
            add('metric_output_changed','Final answer differs from the executed metric output')
    for entry in reversed(result.get('trace',[])):
        if parsed(entry).get('action')!='sql' or 'tool_output' not in entry:continue
        data=entry['tool_output'];rows=data.get('rows',[])
        if data.get('truncated'):
            add('truncated_sql_evidence','The last direct SQL result was truncated','warning')
        if kind in {'number','percent'} and len(rows)==1 and len(rows[0])==1 and type(rows[0][0]) in (int,float):
            # Some valid answers transform a previous result; send this for model
            # review rather than asserting a hard mismatch from one scalar alone.
            if not math.isclose(prediction['answer'],rows[0][0],rel_tol=0,abs_tol=1e-7):
                add('numeric_differs_from_sql','Numeric answer differs from last SQL scalar; check units, percentage and rounding','warning')
        break
    return issues


def gate(prediction,issues):
    return {'type':'abstain'} if any(i['severity']=='hard' for i in issues) else prediction

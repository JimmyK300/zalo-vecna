"""Small verified SQL plans over canonical facts, parameterized by intent.

The model supplies intent/entities, never the arithmetic or SQL for these tools.
"""
import datetime as dt
import json
import re
import calendar
from decimal import Decimal, ROUND_HALF_UP

from .temporal import resolve, event_comparison
from .data import fold

SUPPORTED = ["revenue", "count_orders", "average_order", "top_customers", "top_products", "best_day",
             "customer_group_share", "monthly_trend", "debt", "quantity", "product_revenue",
             "cancel_rate", "revenue_by", "quantity_growth", "revenue_growth", "count_cancelled",
             "debt_customers", "prescription_revenue", "stock_units", "monthly_totals",
             "product_group_share", "sale_price", "purchase_average", "purchase_total", "revenue_share", "prescription_share", "product_group_revenue", "customer_group_revenue"]


def money(value):
    return int(Decimal(str(value or 0)).quantize(Decimal("1"),rounding=ROUND_HALF_UP))


def execute(action, question, sql, layer):
    if 'period' in action:
        nested=action['period']
        if not isinstance(nested,dict) or set(nested)-{'start','end'}:raise ValueError('period must contain only start/end')
        if any(k in action and action[k]!=v for k,v in nested.items()):raise ValueError('Conflicting period and top-level dates')
        action={**{k:v for k,v in action.items() if k!='period'},**nested}
    op=action.get("operation")
    if op not in SUPPORTED: raise ValueError(f"Unsupported operation. Available: {SUPPORTED}")
    extras={"top_customers":{"k","include_values","criterion"},"top_products":{"k","criterion","include_values"},
            "best_day":{"criterion"},"customer_group_share":{"group"},"monthly_trend":{"months","dimension","criterion"},
            "debt":{"as_of","k","by_customer"},"quantity":{"product","unit"},
            "product_revenue":{"product"},"revenue_by":{"dimension","values"},
            "quantity_growth":{"product","unit","compare_start","compare_end"},
            "revenue_growth":{"compare_start","compare_end"},"debt_customers":{"as_of"},
            "prescription_revenue":{"prescription"},"stock_units":{"product","unit","as_of"},
            "monthly_totals":{"months","dimension","criterion"},"product_group_share":{"group"},
            "sale_price":{"product","unit"},"purchase_average":{"product","unit"},
            "revenue_share":{"dimension","values"},"prescription_share":{"prescription"},
            "product_group_revenue":{"group"},"customer_group_revenue":{"group"}}
    unknown=set(action)-{"action","operation","start","end"}-extras.get(op,set())
    if unknown: raise ValueError(f"Unsupported parameters {sorted(unknown)}; they cannot be silently ignored. Use revenue_by for multiple channel/payment totals, or SQL.")
    if op in {'monthly_trend','monthly_totals'} and (action.get('dimension','revenue')!='revenue' or action.get('criterion','revenue')!='revenue'): raise ValueError('Monthly operations support revenue only')
    if op=='top_customers' and action.get('criterion','revenue')!='revenue': raise ValueError('top_customers supports revenue only; use SQL for other rankings')
    if op=='quantity_growth':
        windows=event_comparison(question['question'])
        if windows:
            action={**action,'start':windows[0][0],'end':windows[0][1],
                    'compare_start':windows[1][0],'compare_end':windows[1][1]}
    statements=[]
    def query(statement):
        statements.append(statement)
        return sql.query(statement)["rows"]
    def literal(value):
        return "'"+str(value).replace("'","''")+"'"
    if op in {'purchase_average','purchase_total'}:
        if not layer.get('has_purchase_data'):
            return {'prediction':{'type':'unanswerable','reason':'Dữ liệu phát hành không có sổ/phiếu nhập để tính giá nhập bình quân.'},'sql':[]}
        dates=resolve(question['question'],question['as_of_date'])
        start=action.get('start') or (dates[0] if dates else None)
        end=action.get('end') or (dates[1] if dates else None)
        if dates:start,end=dates
        if not start or not end:raise ValueError('Purchase metrics require start/end')
        dt.date.fromisoformat(start);dt.date.fromisoformat(end)
        if start>end:raise ValueError('start must be <= end')
        if layer.get('purchase_unresolved_dates') or layer.get('purchase_warnings'):
            raise ValueError('Purchase source dates/headers unresolved; inspect raw sources')
        if not layer.get('purchase_date_min'):raise ValueError('No dated purchase facts; inspect sources')
        if end<layer['purchase_date_min'] or start>layer['purchase_date_max']:
            return {'prediction':{'type':'unanswerable','reason':'Kỳ hỏi nằm ngoài khoảng dữ liệu nhập hàng.'},'sql':statements}
        table='purchase_receipts' if op=='purchase_total' else 'purchase_lines'
        condition=f'purchase_date BETWEEN {literal(start)} AND {literal(end)}'
        if query(f"SELECT COUNT(*) FROM {table} WHERE {condition} AND status='unknown'")[0][0]:
            raise ValueError('Unknown purchase status')
        condition+=" AND status<>'cancelled'"
        if op=='purchase_average':
            product=action.get('product')
            if not product:raise ValueError('purchase_average requires product name from data')
            names=query(f'SELECT DISTINCT product_name FROM purchase_lines WHERE fold(product_name)=fold({literal(product)})')
            if not names:raise ValueError('Purchase product not found; search product/catalog names')
            condition+=f' AND fold(product_name)=fold({literal(product)})'
        count,missing,amount=query(f'SELECT COUNT(*),SUM(amount IS NULL),SUM(amount) FROM {table} WHERE {condition}')[0]
        if missing:
            return {'prediction':{'type':'unanswerable','reason':'Nguồn nhập không ghi đủ giá/tiền nhập để tính.'},'sql':statements}
        if op=='purchase_total':answer=money(amount)
        else:
            if not count:raise ValueError('Weighted purchase price undefined without purchases in period')
            unit=action.get('unit')
            rows=query(f'SELECT DISTINCT unit,base_unit,product_id FROM purchase_lines WHERE {condition}')
            if not unit:
                if len({r[0] for r in rows})!=1:raise ValueError('Specify purchase unit when multiple units exist')
                denominator='quantity'
            elif all(fold(unit)==fold(r[0]) for r in rows):denominator='quantity'
            else:
                from .normalized import key
                if fold(unit) in {'base','base_unit','co ban','don vi co ban'} or all(fold(unit)==fold(r[1]) for r in rows):factor=1
                else:
                    ids=','.join(literal(key(r[2])) for r in rows)
                    factors=query(f'SELECT DISTINCT factor FROM unit_conversions WHERE product_key IN ({literal(key(product))},{ids}) AND large_unit={literal(key(unit))}')
                    if len(factors)!=1:raise ValueError('Purchase unit conversion unavailable/ambiguous')
                    factor=factors[0][0]
                denominator=f'base_quantity/{float(factor)}'
            qty,missing_qty=query(f'SELECT SUM({denominator}),SUM(({denominator}) IS NULL) FROM purchase_lines WHERE {condition}')[0]
            if not qty or missing_qty:raise ValueError('Purchase quantities missing or denominator zero')
            answer=money(Decimal(str(amount))/Decimal(str(qty)))
        return {'prediction':{'type':'number','answer':answer},'sql':statements,'period':[start,end]}
    if not layer or not layer.get("date_min"):
        raise ValueError("Sales layer unavailable; inspect typed/raw tables instead")
    dates=resolve(question["question"],question["as_of_date"])
    start=action.get("start") or (dates[0] if dates else None)
    end=action.get("end") or (dates[1] if dates else None)
    text=fold(question['question'])
    if dates and op not in {'quantity_growth','revenue_growth','monthly_trend','debt','debt_customers','stock_units'}:
        # Unambiguous relative periods are authoritative; an LLM cannot widen yesterday to two days.
        if re.search(r'hqua|hom qua|bua qua|hom nay|hnay|bua nay|tuan (?:roi|truoc|nay|ni)|thang (?:nay|ni)|tu dau nam|ngay gan nhat',text):start,end=dates
    if op=='revenue_growth':
        if dates:start,end=dates
        if re.search(r'cung ky nam|nam ngoai|nam truoc',text):
            def previous_year(value):
                d=dt.date.fromisoformat(value);y=d.year-1
                return d.replace(year=y,day=min(d.day,calendar.monthrange(y,d.month)[1])).isoformat()
            action={**action,'compare_start':previous_year(start),'compare_end':previous_year(end)}
        elif re.search(r'thang nay|dau thang nay',text) and re.search(r'th[a-z]* truoc|thang truoc',text):
            from .temporal import context
            a,b=context(question['as_of_date'])['previous_month_same_days']
            action={**action,'compare_start':a,'compare_end':b}
        elif re.search(r'thang (?:lien )?truoc',text) and dates:
            prior=dt.date.fromisoformat(start).replace(day=1)-dt.timedelta(days=1)
            from .temporal import month_range
            a,b=month_range(prior.year,prior.month)
            action={**action,'compare_start':a,'compare_end':b}
    if op in {"debt","debt_customers","stock_units"}:
        end=action.get("as_of") or end or question["as_of_date"]
        start=layer["date_min"]
    if op in {"monthly_trend","monthly_totals"}:
        months=action.get("months")
        if not isinstance(months,list) or len(months)<2: raise ValueError("monthly_trend requires an ordered list of YYYY-MM months")
        parsed=[dt.datetime.strptime(m,"%Y-%m").date() for m in months]
        start=parsed[0].isoformat()
        end=parsed[-1].replace(day=calendar.monthrange(parsed[-1].year,parsed[-1].month)[1]).isoformat()
    if not start or not end: raise ValueError("Specify start/end ISO dates; time context is provided in the question payload")
    dt.date.fromisoformat(start); dt.date.fromisoformat(end)
    if start>end: raise ValueError("start must be <= end")
    if end<layer["date_min"] or start>layer["date_max"]:
        return {"prediction":{"type":"unanswerable","reason":"Kỳ hỏi nằm hoàn toàn ngoài khoảng dữ liệu bán hàng."},"sql":statements}
    period=f"order_date BETWEEN {literal(start)} AND {literal(end)}"
    if op=='stock_units':
        if not layer.get('has_inventory_snapshot') or end!=layer.get('inventory_as_of'):
            raise ValueError('Inventory snapshot unavailable for this date; sales quantity is not remaining stock')
        product=action.get('product');unit=action.get('unit')
        if not product or not unit:raise ValueError('stock_units requires product and large unit')
        rows=query(f'SELECT product_id,stock_quantity,base_unit FROM inventory WHERE fold(product_name)=fold({literal(product)})')
        if len(rows)!=1 or rows[0][1] is None:raise ValueError('Inventory product not uniquely identified')
        from .normalized import key
        conversion=query(f'SELECT factor FROM unit_conversions WHERE product_key IN ({literal(key(product))},{literal(key(rows[0][0]))}) AND large_unit={literal(key(unit))}')
        if not conversion or len({r[0] for r in conversion})!=1:raise ValueError('Large unit conversion unavailable or ambiguous')
        stock=Decimal(str(rows[0][1]));factor=Decimal(str(conversion[0][0]))
        whole=int(stock//factor);remainder=stock-whole*factor
        return {'prediction':{'type':'list','answer':[whole,float(remainder)]},'sql':statements,'period':[start,end]}
    status_table='orders' if layer.get('can_count_orders') else 'sales'
    if op!='sale_price' and layer.get("unknown_status_lines",0) and query(f"SELECT COUNT(*) FROM {status_table} WHERE status='unknown' AND {period}")[0][0]:
        raise ValueError("Unknown statuses remain in sales layer; inspect original status values before computing")
    if layer.get("unresolved_date_rows",0):
        raise ValueError("Some sales dates could not be resolved; inspect sources instead of returning a partial total")
    if op in {'quantity','quantity_growth','top_products'} and layer.get('missing_quantity_rows',0):
        raise ValueError('Unparsed quantities remain; inspect source values before computing quantity/rankings')
    if op not in {'quantity','quantity_growth','count_orders','cancel_rate','sale_price'} and layer.get('missing_amount_rows',0):
        raise ValueError('Unparsed line amounts remain; inspect source values before computing money')
    valid=f"valid_sale=1 AND {period}"
    table="sales" if op in {"top_products","quantity","product_revenue","quantity_growth","prescription_revenue","prescription_share","product_group_share","product_group_revenue","sale_price"} else "orders"
    if table=="orders" and not layer.get("can_count_orders"):
        if op in {"revenue","revenue_growth","monthly_trend","monthly_totals"}: table="sales"
        else: raise ValueError("Order identifiers unavailable; do not invent order counts")
    if table=="sales" and not layer.get("has_sales_lines"):
        raise ValueError("This business has no product-line details")
    if op in {"product_revenue","quantity","quantity_growth"}:
        product=action.get("product")
        if product:
            values=query(f"SELECT DISTINCT product_name FROM sales WHERE fold(product_name)=fold({literal(product)})")
            if not values: raise ValueError("Product name not found; search product names before calculating")
            valid+=f" AND fold(product_name)=fold({literal(product)})"
        elif op=="product_revenue": raise ValueError("product_revenue requires a product name")
        unit=action.get("unit")
        if unit and fold(unit) not in {"base","base_unit","co ban","don vi co ban"}:
            units=query("SELECT DISTINCT base_unit FROM sales WHERE base_unit IS NOT NULL")
            if not any(fold(r[0])==fold(unit) for r in units): raise ValueError("Requested base unit not found; inspect actual units")
            valid+=f" AND fold(base_unit)=fold({literal(unit)})"
    if op=='sale_price':
        product=action.get('product')
        if not product:raise ValueError('sale_price requires a product name')
        rows=query(f'SELECT DISTINCT unit_price*quantity/base_quantity,base_unit,product_id FROM sales WHERE {period} AND fold(product_name)=fold({literal(product)}) AND base_quantity>0 AND unit_price IS NOT NULL')
        if len(rows)!=1 or rows[0][0] is None:raise ValueError('No unique unit sale price; inspect source prices and units')
        price=Decimal(str(rows[0][0]));unit=action.get('unit')
        if unit and fold(unit) not in {'base','base_unit','co ban','don vi co ban',fold(rows[0][1])}:
            from .normalized import key
            factors=query(f'SELECT factor FROM unit_conversions WHERE product_key IN ({literal(key(product))},{literal(key(rows[0][2]))}) AND large_unit={literal(key(unit))}')
            if not factors or len({r[0] for r in factors})!=1:raise ValueError('Requested sale unit conversion missing/ambiguous')
            price*=Decimal(str(factors[0][0]))
        result={'type':'number','answer':money(price)}
    elif op in {'prescription_revenue','prescription_share'}:
        if not layer.get('has_prescription_attributes'):raise ValueError('Prescription attributes unavailable')
        flag=int(action.get('prescription',True))
        amount=query(f'SELECT SUM(s.line_amount) FROM sales s WHERE {valid} AND COALESCE((SELECT p.prescription FROM product_attributes p WHERE p.product_id=s.product_id),0)={flag}')[0][0]
        if op=='prescription_share':
            total=query(f'SELECT SUM(line_amount) FROM sales WHERE {valid}')[0][0]
            if not total:raise ValueError('Prescription share denominator zero')
            result={'type':'percent','answer':float((Decimal(str(amount or 0))*100/Decimal(str(total))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))}
        else:result={'type':'number','answer':money(amount)}
    elif op in {"cancel_rate","count_cancelled"}:
        total,cancelled=query(f"SELECT COUNT(*),SUM(status='cancelled') FROM orders WHERE {period}")[0]
        if not total: raise ValueError("Cancellation percentage denominator is zero")
        result=({'type':'number','answer':cancelled or 0} if op=='count_cancelled' else {"type":"percent","answer":float((Decimal(cancelled or 0)*100/total).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))})
    elif op in {"revenue_by","revenue_share"}:
        dimension=action.get("dimension"); values=action.get("values")
        if dimension not in {"channel","payment_method","customer_group"} or not isinstance(values,list) or not 1<=len(values)<=20:
            raise ValueError("revenue_by requires dimension channel/payment_method/customer_group and ordered values list")
        aliases={"tai quay":"store","quay":"store","zalo oa":"zalo_oa","tien mat":"cash","chuyen khoan":"transfer","ghi no":"debt","vi dien tu":"ewallet"}
        found=query(f"SELECT DISTINCT {dimension} FROM orders WHERE {dimension} IS NOT NULL")
        known={fold(r[0]):r[0] for r in found}
        amounts=[]
        resolved_values=[]
        for value in values:
            normalized=fold(value)
            if normalized not in known:
                target=aliases.get(normalized,normalized)
                candidates=[v for k,v in known.items() if aliases.get(k,k)==target]
                if len(candidates)!=1:raise ValueError(f"Unknown/ambiguous {dimension} {value}; inspect distinct values")
                normalized=fold(candidates[0])
            resolved_values.append(normalized)
            amounts.append(money(query(f"SELECT SUM(revenue) FROM orders WHERE {valid} AND fold({dimension})=fold({literal(known[normalized])})")[0][0]))
        if op=='revenue_share':
            if len(set(resolved_values))!=len(values):raise ValueError('Share values must be distinct after alias resolution')
            total=query(f'SELECT SUM(revenue) FROM orders WHERE {valid}')[0][0]
            if not total:raise ValueError('Share denominator zero')
            result={'type':'percent','answer':float((Decimal(sum(amounts))*100/Decimal(str(total))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))}
        else:result={"type":"list","answer":amounts}
    elif op in {"quantity_growth","revenue_growth"}:
        a,b=action.get("compare_start"),action.get("compare_end")
        if not a or not b: raise ValueError("quantity_growth needs compare_start/compare_end for the baseline period")
        dt.date.fromisoformat(a);dt.date.fromisoformat(b)
        if a>b: raise ValueError("Comparison start must precede end")
        if b<layer['date_min'] or a>layer['date_max']: raise ValueError("Baseline period lies outside data")
        previous=valid.replace(period,f"order_date BETWEEN {literal(a)} AND {literal(b)}",1)
        measure='base_quantity' if op=='quantity_growth' else 'line_amount' if table=='sales' else 'revenue'
        current=query(f"SELECT SUM({measure}) FROM {table} WHERE {valid}")[0][0] or 0
        before=query(f"SELECT SUM({measure}) FROM {table} WHERE {previous}")[0][0] or 0
        if not before: raise ValueError("Growth denominator is zero")
        value=((Decimal(str(current))-Decimal(str(before)))*100/Decimal(str(before))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
        result={"type":"percent","answer":float(value)}
    elif op in {"revenue","product_revenue"}:
        col="line_amount" if table=="sales" else "revenue"
        result={"type":"number","answer":money(query(f"SELECT SUM({col}) FROM {table} WHERE {valid}")[0][0])}
    elif op=="quantity":
        value=query(f"SELECT SUM(base_quantity) FROM sales WHERE {valid}")[0][0] or 0
        result={"type":"number","answer":round(value,2)}
    elif op=="count_orders":
        result={"type":"number","answer":query(f"SELECT COUNT(*) FROM orders WHERE {valid}")[0][0]}
    elif op=="average_order":
        amount,count=query(f"SELECT SUM(revenue),COUNT(*) FROM orders WHERE {valid}")[0]
        if not count: raise ValueError("Average order is undefined without valid orders")
        result={"type":"number","answer":money(Decimal(str(amount))/count)}
    elif op in {"top_customers","top_products"}:
        k=int(action.get("k",1))
        if not 1<=k<=20: raise ValueError("k must be between 1 and 20")
        if op=="top_customers":
            rows=query(f"SELECT COALESCE(MAX(customer_name),customer_id),SUM(revenue) total FROM orders WHERE {valid} AND customer_id IS NOT NULL GROUP BY customer_id ORDER BY total DESC,customer_id ASC LIMIT {k}")
        else:
            measure="line_amount" if action.get("criterion")=="revenue" else "base_quantity"
            rows=query(f"SELECT MAX(product_name),SUM({measure}) total FROM sales WHERE {valid} AND fold(product_name)<>'phi phuc vu' GROUP BY COALESCE(product_id,fold(product_name)) ORDER BY total DESC,COALESCE(product_id,fold(product_name)) ASC LIMIT {k}")
        # The question determines output shape, not an often over-eager model flag.
        text=fold(question["question"])
        include_values=bool(re.search(r"moi.*(?:bao nhieu|bnhiu|nhieu)|kem.*(?:tien|so luong|doanh thu|gia tri)|bang\s+(?:\d+\s+)?khach",text))
        if include_values:
            result={"type":"table","answer":{"columns":["khach_hang" if op=="top_customers" else "san_pham","gia_tri"],"rows":[[r[0],money(r[1])] for r in rows]}}
        elif k==1:
            if not rows: raise ValueError("No ranking candidate in period")
            result={"type":"string","answer":rows[0][0]}
        else: result={"type":"list","answer":[r[0] for r in rows]}
    elif op=="best_day":
        measure="COUNT(*)" if action.get("criterion")=="orders" else "SUM(revenue)"
        rows=query(f"SELECT order_date,{measure} total FROM orders WHERE {valid} GROUP BY order_date ORDER BY total DESC,order_date ASC LIMIT 1")
        if not rows: raise ValueError("No candidate date")
        result={"type":"date","answer":rows[0][0]}
    elif op in {"customer_group_share","product_group_share","customer_group_revenue","product_group_revenue"}:
        group=action.get("group")
        if not group: raise ValueError("Specify customer group from data")
        field='customer_group' if op.startswith('customer') else 'product_group'
        groups=query(f"SELECT DISTINCT {field} FROM {table} WHERE {field} IS NOT NULL")
        if not groups: raise ValueError("Customer-group information is absent; do not infer from names")
        if not any(fold(r[0])==fold(group) for r in groups):raise ValueError('Requested group not found; inspect exact group names instead of returning zero')
        amount='revenue' if table=='orders' else 'line_amount'
        total,part=query(f"SELECT SUM({amount}),SUM(CASE WHEN fold({field})=fold({literal(group)}) THEN {amount} ELSE 0 END) FROM {table} WHERE {valid}")[0]
        if op.endswith('revenue'):result={'type':'number','answer':money(part)}
        else:
            if not total: raise ValueError("Percentage denominator is zero")
            value=(Decimal(str(part or 0))*100/Decimal(str(total))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
            result={"type":"percent","answer":float(value)}
    elif op in {"monthly_trend","monthly_totals"}:
        amount='revenue' if table=='orders' else 'line_amount'
        rows=query(f"SELECT substr(order_date,1,7),SUM({amount}) FROM {table} WHERE {valid} GROUP BY substr(order_date,1,7)")
        series=dict(rows)
        values=[series.get(m,0) for m in months]
        result=({'type':'list','answer':[money(v) for v in values]} if op=='monthly_totals' else {"type":"boolean","answer":all(b>a for a,b in zip(values,values[1:]))})
    else:
        if not layer.get("has_payment_events"):
            raise ValueError("Historical debt requires payment event history; a final paid snapshot is insufficient")
        limit=int(action.get("k") or 1)
        if not 1<=limit<=20: raise ValueError("k must be between 1 and 20")
        debt=f"""WITH paid AS (SELECT order_id,SUM(amount) amount FROM payments WHERE payment_date<={literal(end)} GROUP BY order_id),
        debt AS (SELECT o.*, MAX(COALESCE(amount_due,0)-COALESCE(p.amount,0),0) balance FROM orders o LEFT JOIN paid p USING(order_id)
        WHERE o.order_date<={literal(end)} AND o.status<>'cancelled' AND fold(COALESCE(payment_method,'')) NOT LIKE '%cod%'
        AND fold(COALESCE(channel,'')) NOT LIKE '%shopee%' AND fold(COALESCE(channel,'')) NOT LIKE '%tmdt%')"""
        if op=='debt_customers':
            result={'type':'number','answer':query(debt+' SELECT COUNT(DISTINCT customer_id) FROM debt WHERE customer_id IS NOT NULL AND balance>0')[0][0]}
        elif action.get("by_customer",True):
            rows=query(debt+f" SELECT COALESCE(MAX(customer_name),customer_id),SUM(balance) total FROM debt WHERE customer_id IS NOT NULL AND balance>0 GROUP BY customer_id ORDER BY total DESC,customer_id ASC LIMIT {limit}")
            result={"type":"table","answer":{"columns":["khach_hang","so_tien_no"],"rows":[[r[0],money(r[1])] for r in rows]}}
        else: result={"type":"number","answer":money(query(debt+" SELECT SUM(balance) FROM debt")[0][0])}
    return {"prediction":result,"sql":statements,"period":[start,end]}

"""Purchase facts from source schemas, independent of question/reference labels."""
import re

from .data import fold, iso_date, number


FIELDS = {
    'id': 'ma_nhap_hang ma_don_nhap ma_phieu_nhap so_phieu',
    'date': 'ngay_nhap ngay_nhap_kho thoi_gian ngay date',
    'product_id': 'ma_hang ma_hang_hoa ma_sku ma_sp product_id',
    'product_name': 'ten_mat_hang ten_hang ten_san_pham product hang',
    'quantity': 'sl_nhap so_luong_nhap so_luong sl quantity',
    'unit': 'dvt_nhap don_vi_tinh dvt don_vi unit',
    'price': 'don_gia_nhap gia_nhap don_gia unit_price',
    'amount': 'tong_gia_tri_nhap can_tra_ncc thanh_tien tong_tien amount',
    'status': 'trang_thai tinh_trang status',
    'supplier': 'ma_nha_cung_cap ma_ncc nha_cung_cap supplier',
}


def create_purchases(con, tables, info, product_index, unit_index):
    from .normalized import key, quantity_number
    receipts, lines, warnings = {}, [], []
    inline_suppliers={}
    source_tables = []
    unresolved = 0

    def mapping(columns):
        compact = {c.replace('_', ''): c for c in columns}
        return {role: next((compact[a.replace('_', '')] for a in aliases.split()
                            if a.replace('_', '') in compact), None)
                for role, aliases in FIELDS.items()}

    def value(row, m, role):
        return row.get(m[role]) if m[role] else None

    def state(raw):
        if raw is None or not str(raw).strip(): return 'received'
        text=fold(raw)
        if re.search(r'\bhuy\b|cancel', text): return 'cancelled'
        if re.search(r'nhap|received|completed|hoan thanh', text): return 'received'
        return 'unknown'

    candidates=[]
    for name, rows in tables.items():
        source=fold(' '.join(s['file']+' '+s['sheet'] for s in info['tables'][name]['sources']))
        if not re.search(r'nhap|purchase', source): continue
        m=mapping(info['tables'][name]['columns'])
        if not m['id']: continue
        if not m['date'] and not m['quantity']: continue
        source_tables.append(name)
        candidates.append((name, rows, m))
    # Header totals are one per receipt, not repeated per detail line.
    for name, rows, m in candidates:
        if m['quantity']: continue
        for row in rows:
            rid=value(row,m,'id')
            if rid is None: continue
            date=iso_date(value(row,m,'date'),row.get('_source_year'))
            if not date: unresolved+=1; continue
            item=[str(rid).strip(),date,value(row,m,'supplier'),state(value(row,m,'status')),
                  number(value(row,m,'amount')),name,row['_source_row']]
            if item[0] in receipts and receipts[item[0]][:5]!=item[:5]:
                warnings.append('Conflicting purchase receipt headers: '+item[0])
            receipts[item[0]]=item
    seen=set()
    for name, rows, m in candidates:
        if not m['quantity']: continue
        carried={}
        for original in rows:
            row=dict(original)
            rid=value(row,m,'id')
            if rid is None: continue
            rid=str(rid).strip()
            prior=carried.setdefault((row.get('_source_file'),row.get('_source_sheet'),rid),{})
            for role in ['date','supplier','status']:
                col=m[role]
                if not col: continue
                if row.get(col) not in (None,''):prior[col]=row[col]
                elif col in prior:row[col]=prior[col]
            header=receipts.get(rid)
            supplier=value(row,m,'supplier')
            if supplier not in (None,''):
                if rid in inline_suppliers and inline_suppliers[rid]!=supplier:
                    warnings.append('Conflicting purchase supplier values: '+rid)
                inline_suppliers[rid]=supplier
            date=header[1] if header else iso_date(value(row,m,'date'),row.get('_source_year'))
            if not date:unresolved+=1; continue
            signature=(rid,tuple(original[c] for c in info['tables'][name]['columns']))
            if signature in seen:continue
            seen.add(signature)
            pid=value(row,m,'product_id'); title=value(row,m,'product_name')
            catalog=product_index.get(('id',str(pid).strip())) if pid else None
            catalog=catalog or product_index.get(('name',key(title)))
            base_unit=None
            if catalog:
                pr,pm=catalog
                pid=pid or pr.get(pm.get('product_id'))
                title=pr.get(pm.get('product_name')) or title
                base_unit=pr.get(pm.get('base_unit'))
            unit=value(row,m,'unit')
            qty=quantity_number(value(row,m,'quantity'),unit)
            price=number(value(row,m,'price')); amount=number(value(row,m,'amount'))
            if amount is None and price is not None and qty is not None: amount=price*qty
            conversion=unit_index.get((key(pid),key(unit))) or unit_index.get((key(title),key(unit)))
            base_qty=qty*(conversion[0] if conversion else 1) if qty is not None else None
            base_unit=(conversion[1] if conversion else None) or base_unit or unit
            st=header[3] if header else state(value(row,m,'status'))
            lines.append([rid,date,pid,title,qty,unit,base_qty,base_unit,price,amount,st,name,original['_source_row']])
    grouped={}
    for line in lines:
        grouped.setdefault(line[0],[]).append(line)
    for rid, items in grouped.items():
        total=sum(r[9] for r in items) if all(r[9] is not None for r in items) else None
        if rid not in receipts:
            r=items[0]
            receipts[rid]=[rid,r[1],inline_suppliers.get(rid),r[10],total,r[11],r[12]]
        elif receipts[rid][4] is None:receipts[rid][4]=total
    line_cols=['receipt_id','purchase_date','product_id','product_name','quantity','unit','base_quantity',
               'base_unit','unit_price','amount','status','_source_table','_source_row']
    receipt_cols=['receipt_id','purchase_date','supplier','status','amount','_source_table','_source_row']
    for name, cols, rows in [('purchase_lines',line_cols,lines),('purchase_receipts',receipt_cols,list(receipts.values()))]:
        con.execute(f'CREATE TABLE {name} ({",".join(cols)})')
        con.executemany(f'INSERT INTO {name} VALUES ({",".join("?" for _ in cols)})',rows)
        con.execute(f'CREATE INDEX {name}_date ON {name}(purchase_date)')
        info['tables'][name]={'columns':cols,'rows':len(rows),'sources':[{'file':'derived','sheet':'purchase mappings'}],
                              'sample':rows[:2]}
    dates=[r[1] for r in receipts.values()]
    return {'has_purchase_data':bool(source_tables),'purchase_tables':source_tables,
            'purchase_date_min':min(dates) if dates else None,'purchase_date_max':max(dates) if dates else None,
            'purchase_unresolved_dates':unresolved,'purchase_warnings':warnings}

"""Typed raw tables and a conservative, data-driven sales layer.

Mappings depend on headers, README descriptions and table structure, never on
business/question identifiers. Original cells remain in raw_* tables.
"""
import functools
import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from .data import fold, identifier, iso_date, number, quote

VERSION = "typed-sales-v13"
parse_number = functools.lru_cache(maxsize=100000)(number)
parse_date = functools.lru_cache(maxsize=100000)(iso_date)

ALIASES = {
 "order_id": "ma_hoa_don ma_don_hang ma_don ma_bill ma_chung_tu order_id so_ct so_phieu so_hd ma_giao_dich don_so stt",
 "date": "ngay_dat_hang ngay_ban ngay ngay_giao_dich ngay_thang ngay_mua thoi_gian ngay_lap ngay_dat date order_date",
 "status": "trang_thai_don_hang trang_thai tinh_trang_don tinh_trang ket_qua status ghi_chu_tt xu_ly trang_thai_xu_ly",
 "product_id": "ma_hang ma_hang_hoa ma_sp ma_sku ma_hh ma_hang_ban id_sp sku product_id ma_mon",
 "product_name": "ten_hang ten_hang_hoa ten_san_pham ten_mat_hang hang mat_hang noi_dung ten_mon mon san_pham chi_tiet_hang hang_hoa product ten_goi",
 "quantity": "so_luong so_luong_san_pham sl_ban slg s_l sl soluong qty",
 "base_quantity": "so_luong_quy_doi quy_ra_dv_le sl_co_ban so_luong_le sl_quy_doi sl_quy_ra_le",
 "line_amount": "thanh_tien tien_ban tri_gia t_tien cong tong_tien amount line_amount",
 "unit_price": "don_gia gia_san_pham gia_ban gia_le don_gia_ban gia_1_dv gia_niem_yet gia_don_vi gia_ban_le",
 "unit_cost": "gia_von_luc_ban gia_von cost_price",
 "unit": "don_vi_tinh dvt unit donvi dv_tinh d_vi don_vi",
 "base_unit": "don_vi_co_ban don_vi_co_so don_vi_goc dv_nho dv_goc dvt don_vi_tinh don_vi",
 "customer_id": "ma_kh ma_khach_hang ma_kh_mua ma_khach_mua so_kh customer_id ma_khach ma_khach_le_si ma_kh_mua",
 "customer_name": "ten_khach_hang khach_hang ten_kh khach doi_tac ben_mua khach_le_si kh ten_goi",
 "customer_group": "nhom_khach_hang nhom nhom_khach hang loai_khach_hang loai_khach loai_hinh_khach doi_tuong_kh",
 "product_group": "nhom_hang_3_cap nhom_hang_hoa nhom_hang loai_san_pham phan_loai nhom danh_muc",
 "revenue": "tong_tien_hang tien_hang_truoc_giam tien_hang revenue",
 "due": "tong_sau_giam_gia tong_thanh_toan phai_thu tong_bill tong_tien tongtien",
 "paid": "khach_da_tra da_thu da_thanh_toan da_tra thanh_toan khach_tra",
 "payment_amount": "so_tien_thu tien_thu gia_tri tien so_tien",
 "payment_date": "thoi_gian_thu thoi_diem thoi_gian ngay_thu ngay_tt ngay_tra thoi_diem_thu ngay_gio_thu",
 "payment_method": "phuong_thuc_thanh_toan phuong_thuc_tt hinh_thuc_thanh_toan pttt phuong_thuc_tt phuong_thuc loai_tt",
 "channel": "kenh_ban kenh_ban_hang nguon_don ban_qua kenh channel",
 "employee": "nguoi_ban ma_nv_ban ma_nhan_vien nhan_vien tho employee_id",
 "factor": "ti_le_quy_doi ti_le he_so he_so_quy_doi he_so_qd quy_doi bang",
 "large_unit": "don_vi_quy_doi dv_lon dv_nhap don_vi dvt don_vi_tinh",
 "stock": "ton_cuoi_ky ton_thuc ton_cuoi ton_kho",
 "prescription": "thuoc_ke_don ban_theo_don ke_don",
}
ALIASES = {key: values.split() for key, values in ALIASES.items()}
DESCRIPTIONS = {
 "order_id": ("ma don",), "date": ("ngay ban",), "status": ("trang thai don",),
 "product_id": ("ma hang",), "product_name": ("ten hang",),
 "quantity": ("so luong",), "base_quantity": ("so luong da doi",),
 "line_amount": ("thanh tien sau giam",), "unit_price": ("don gia",),
 "unit": ("don vi tinh",), "customer_id": ("ma khach",),
 "customer_name": ("ten khach",), "product_group": ("nhom hang",),
 "payment_method": ("hinh thuc thanh toan cua don",),
 "payment_amount": ("so tien tra",), "payment_date": ("thoi diem thu",),
 "base_unit": ("don vi co ban",), "large_unit": ("don vi lon",), "factor": ("he so:",),
 "stock": ("ton kho",), "prescription": ("thuoc ke don",),
}


def field(columns, role, descriptions):
    # A README data dictionary overrides lexical guesses when the meaning is explicit.
    for col in columns:
        description = descriptions.get(col, "")
        if any(description.startswith(prefix) for prefix in DESCRIPTIONS.get(role, ())):
            if role == "quantity" and description.startswith("so luong da doi"):
                continue
            return col
    exact=next((key for key in ALIASES.get(role, []) if key in columns), None)
    if exact: return exact
    compact={c.replace('_',''):c for c in columns}
    return next((compact[key.replace('_','')] for key in ALIASES.get(role, []) if key.replace('_','') in compact),None)


def status(value):
    text = fold(value or "").strip()
    if re.search(r"huy|cancel|khong giao duoc", text): return "cancelled"
    if re.search(r"\btra\b|hoan hang|returned|return", text) or text == "hoan": return "returned"
    if re.search(r"dang giao|cho giao|dang gui|dang van chuyen|dang xu ly|shipping|delivery", text): return "shipping"
    if re.search(r"xong|hoan thanh|thanh cong|da lay|da giao",text) or text in {"completed","complete","ok","da thanh toan"}: return "completed"
    return "unknown"


def key(value):
    return " ".join(re.sub(r"[^\w\s]", " ", fold(value or "")).split())


def quantity_number(value, unit):
    parsed=parse_number(value)
    if parsed is not None: return parsed
    text=fold(value or '').strip(); suffix=fold(unit or '').strip()
    if suffix and text.endswith(suffix):
        return parse_number(text[:-len(suffix)].strip())
    return None


def customer_key(code, name):
    if code is not None and str(code).strip(): return str(code).strip()
    if key(name) in {"", "le", "kl", "k l", "khach le", "khach vang lai", "khach mua le", "khach khong luu"}: return None
    return key(name)


def kind_for_column(col):
    if col.startswith("_source") or re.match(r"^(ma_|id|sku|code|sdt|dien_thoai|so_dien_thoai|phone|imei|barcode|customer_id|order_id)", col):
        return "raw"
    if col == "ngay" or col.startswith("ngay_") or col.replace('_','') in {s.replace('_','') for s in ALIASES['date']+ALIASES['payment_date']+["hsd","han_dung","han_thanh_toan"]}:
        return "date"
    numeric = {v for k, vs in ALIASES.items() if k in {"quantity", "base_quantity", "line_amount", "unit_price", "unit_cost", "revenue", "due", "paid", "payment_amount", "factor"} for v in vs}
    if col.replace('_','') in {s.replace('_','') for s in numeric} or re.search(r"^(ton_|sl_|so_luong|gia_|tien_|chi_phi|giam_gia|con_no|du_no)", col):
        return "number"
    return "raw"


def build_normalized(raw_path, manifest, readme, cache_dir):
    started = time.perf_counter()
    output = Path(cache_dir) / (manifest["fingerprint"] + "_" + VERSION + ".sqlite")
    info_path = output.with_suffix(".json")
    if output.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return output, info, {"typed_cache_hit": True, "normalization_seconds": time.perf_counter()-started}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): output.unlink()
    source = sqlite3.connect(raw_path)
    con = sqlite3.connect(output)
    source.backup(con)
    source.close()
    descriptions = {identifier(m[1]): fold(m[2]).strip() for m in re.finditer(r"\*\*([^*]+)\*\*\s*:\s*([^\n]+)", readme)}
    info = json.loads(json.dumps(manifest))
    info["normalization_version"] = VERSION
    info["warnings"] = ["Typed tNN tables retain source rows. Original cells are in raw_tNN.",
                         "Use sales/orders for supported sales metrics. Do not count lines as orders.",
                         "Missing capabilities are explicit; unsupported metrics require raw-table investigation."]
    table_rows, mappings = {}, {}
    try:
        for name, meta in manifest["tables"].items():
            if name == "calendar": continue
            cursor = con.execute(f"SELECT * FROM {quote(name)}")
            cols = [c[0] for c in cursor.description]
            kinds = {c: kind_for_column(c) for c in meta["columns"]}
            raw = cursor.fetchall()
            typed, failures = [], defaultdict(int)
            for values in raw:
                row = dict(zip(cols, values))
                for c, kind in kinds.items():
                    old = row[c]
                    if old is None or old == "": continue
                    converted = parse_date(old, row.get("_source_year")) if kind == "date" else parse_number(old) if kind == "number" else old
                    if converted is None:
                        failures[c] += 1
                    else:
                        row[c] = converted
                typed.append(row)
            # A mostly textual field that happens to share a numeric alias is not coerced.
            successful_types = {}
            for c, kind in kinds.items():
                nonempty = sum(v[cols.index(c)] not in (None, "") for v in raw)
                if nonempty and failures[c] > nonempty * .05:
                    for row, values in zip(typed, raw): row[c] = values[cols.index(c)]
                    successful_types[c] = "raw (ambiguous or parse failures)"
                else: successful_types[c] = kind
            con.execute(f"ALTER TABLE {quote(name)} RENAME TO {quote('raw_' + name)}")
            con.execute(f"CREATE TABLE {quote(name)} ({','.join(quote(c) for c in cols)})")
            con.executemany(f"INSERT INTO {quote(name)} VALUES ({','.join('?' for _ in cols)})", ([r[c] for c in cols] for r in typed))
            table_rows[name] = typed
            meta_out = info["tables"][name]
            meta_out["column_types"] = successful_types
            meta_out["parse_failures"] = dict(failures)
            meta_out["sample"] = [[r[c] for c in meta["columns"]] for r in typed[:2]]
            mappings[name] = {role: field(meta["columns"], role, descriptions) for role in ALIASES}
        create_sales(con, table_rows, mappings, info, readme)
        con.commit()
        info_path.write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    finally:
        con.close()
    return output, info, {"typed_cache_hit": False,"normalization_seconds":time.perf_counter()-started}


def create_sales(con, tables, mappings, info, readme):
    def val(row, mapping, role):
        col = mapping.get(role)
        return row.get(col) if col else None

    def sources(name):
        return fold(" ".join(s["file"] + " " + s["sheet"] for s in info["tables"][name]["sources"]))

    line_tables = [name for name,m in mappings.items() if (m["product_name"] or m["product_id"]) and m["quantity"] and m["line_amount"]
                   and not re.search(r"nhap|purchase|giao.hang|delivery", sources(name))]
    header_tables = [name for name,m in mappings.items() if m["order_id"] and m["date"] and not m["quantity"]
                     and (m["revenue"] or m["due"] or m["paid"])
                     and not re.search(r"nhap|purchase|phieu.thu|so.quy|payment|thanh.toan", sources(name))]
    customer_tables = [name for name,m in mappings.items() if m["customer_id"] and m["customer_name"] and not m["order_id"]
                       and not m["quantity"]]
    product_tables = [name for name,m in mappings.items() if m["product_name"] and not m["order_id"] and not m["quantity"]
                      and re.search(r"danh|san.?pham|hang.?hoa|product", sources(name))]
    customer_index, product_index = {}, {}
    for name in customer_tables:
        m=mappings[name]
        for row in tables[name]:
            code=val(row,m,"customer_id")
            if code: customer_index[str(code).strip()] = row,m
    for name in product_tables:
        m=mappings[name]
        for row in tables[name]:
            code, title=val(row,m,"product_id"),val(row,m,"product_name")
            if code: product_index[("id",str(code).strip())] = row,m
            if title: product_index.setdefault(("name",key(title)),(row,m))
    headers = {}
    for name in header_tables:
        m=mappings[name]
        for row in tables[name]:
            code=val(row,m,"order_id")
            if code is not None: headers[str(code).strip()] = row,m

    unit_index={}
    for name,m in mappings.items():
        if not m["factor"] or not m["large_unit"]: continue
        for row in tables[name]:
            code=val(row,m,"product_id") or val(row,m,"product_name")
            factor=parse_number(val(row,m,"factor"))
            if code and factor:
                small=val(row,m,"base_unit") if m['base_unit']!=m['large_unit'] else None
                unit_index[(key(code),key(val(row,m,"large_unit")))] = (factor,small)
    # Unit conversions documented inline in a business README are data, not assumptions.
    for match in re.finditer(r"(?m)^\s*-\s+([^:\n]+):\s*1\s+([^=\n]+?)\s*=\s*(\d+(?:[.,]\d+)?)\s+(\S+)",readme):
        title,large,factor,small=match.groups()
        unit_index[(key(title),key(large))]=(number(factor),small)

    columns = ["order_id","order_date","customer_id","customer_name","customer_group","product_id","product_name",
               "product_group","quantity","base_quantity","unit","base_unit","line_amount","unit_price","unit_cost",
               "status","valid_sale","payment_method","channel","employee","_source_table","_source_row"]
    sales=[]
    states=defaultdict(set)
    due_by_order, paid_by_order = {},{}
    seen=set()
    has_order_ids=False
    negative_returns=bool(re.search(r'so luong am',fold(readme)))
    dropped_dates=0
    for name in line_tables:
        m=mappings[name]
        carried={}
        previous_order=None
        previous_source=None
        previous_date=None
        for original in tables[name]:
            row=dict(original)
            # Spreadsheet totals/notes are not transactions or continuation lines.
            title=val(row,m,'product_name') or val(row,m,'product_id')
            if (not title and val(row,m,'quantity') in (None,'')) or (
                not val(row,m,'date') and val(row,m,'quantity') in (None,'') and
                (key(title) in {'cong','tong','tong cong','subtotal','total'} or re.match(r'^(tong ngay|tong thang|ghi chu)',key(title)))):
                continue
            current_source=(row.get("_source_file"),row.get("_source_sheet"))
            if current_source!=previous_source:
                previous_order=None
                previous_date=None
                previous_source=current_source
            if m.get('date'):
                if val(row,m,'date') not in (None,''):previous_date=val(row,m,'date')
                elif not m.get('order_id') and previous_date is not None:row[m['date']]=previous_date
            raw_id=val(row,m,"order_id")
            # Exported continuation rows omit order fields; never fill across source boundaries.
            if m.get("order_id") and raw_id in (None,"") and val(row,m,"date") in (None,"") and previous_order:
                raw_id=previous_order
                row[m["order_id"]]=raw_id
            order_id=str(raw_id).strip() if raw_id not in (None,"") else None
            if order_id: previous_order=order_id
            if order_id:
                previous=carried.setdefault(order_id,{})
                for role in ["date","status","customer_id","customer_name","due","paid","payment_method","channel","employee"]:
                    col=m.get(role)
                    if not col: continue
                    if row.get(col) not in (None,""): previous[col]=row[col]
                    elif col in previous: row[col]=previous[col]
            if order_id and order_id in headers:
                header,hm=headers[order_id]
            else: header,hm=row,m
            date=val(header,hm,"date") or val(row,m,"date")
            date=parse_date(date,header.get("_source_year"))
            if not date:
                dropped_dates+=1
                continue
            has_order_ids |= order_id is not None
            # Without an order identifier, preserve every independent raw record.
            identity=(order_id,date) if order_id else (name,original["_source_row"],original["_source_sheet"])
            if order_id:
                signature=(identity,tuple(original.get(c) for c in info["tables"][name]["columns"]))
                if signature in seen: continue
                seen.add(signature)
            order_key=order_id if order_id else None
            st=status(val(header,hm,"status") or val(row,m,"status"))
            states[order_id or identity].add(st)
            cid=val(header,hm,"customer_id") or val(row,m,"customer_id")
            cname=val(header,hm,"customer_name") or val(row,m,"customer_name")
            group=None
            if cid is not None and str(cid).strip() in customer_index:
                cr,cm=customer_index[str(cid).strip()]
                cname=val(cr,cm,"customer_name") or cname
                group=val(cr,cm,"customer_group")
            cid=customer_key(cid,cname)
            pid=val(row,m,"product_id")
            pname=val(row,m,"product_name")
            catalog=product_index.get(("id",str(pid).strip())) if pid else None
            catalog=catalog or product_index.get(("name",key(pname)))
            pgroup,base_unit=None,None
            if catalog:
                pr,pm=catalog
                pid=pid or val(pr,pm,"product_id")
                pname=val(pr,pm,"product_name") or pname
                pgroup=val(pr,pm,"product_group")
                base_unit=val(pr,pm,"base_unit")
            unit=val(row,m,"unit")
            qty=quantity_number(val(row,m,"quantity"),unit)
            if negative_returns and qty is not None and qty<0: states[order_id or identity].add('returned')
            base_qty=quantity_number(val(row,m,"base_quantity"),base_unit or unit)
            conversion=unit_index.get((key(pid),key(unit))) or unit_index.get((key(pname),key(unit)))
            if base_qty is None and qty is not None: base_qty=qty*(conversion[0] if conversion else 1)
            if conversion: base_unit=conversion[1] or base_unit
            base_unit=base_unit or unit
            amount=parse_number(val(row,m,"line_amount"))
            due_by_order[identity]=parse_number(val(header,hm,"due"))
            paid_by_order[identity]=parse_number(val(header,hm,"paid"))
            data=[order_key,date,cid,cname,group,pid,pname,pgroup,qty,base_qty,unit,base_unit,amount,
                  parse_number(val(row,m,"unit_price")),parse_number(val(row,m,"unit_cost")),st,None,
                  val(header,hm,"payment_method"),val(header,hm,"channel"),val(header,hm,"employee"),name,original["_source_row"]]
            sales.append((identity,data))
    for identity,row in sales:
        st=states[row[0] or identity]
        row[15]=next((s for s in ["cancelled","returned","shipping","completed"] if s in st),"unknown")
        row[16]=int(row[15] in {"completed","shipping"})
    con.execute(f"CREATE TABLE sales ({','.join(quote(c) for c in columns)})")
    con.executemany(f"INSERT INTO sales VALUES ({','.join('?' for _ in columns)})",(r for _,r in sales))
    con.execute("CREATE INDEX sales_date ON sales(order_date)")
    con.execute("CREATE INDEX sales_order ON sales(order_id)")
    # Header-only businesses can still expose order revenue but cannot support product questions.
    order_rows={}
    for identity,row in sales:
        if row[0] is None: continue
        # Return events may repeat the same order on a later date. They do not create a second order.
        target=order_rows.setdefault(row[0],{"order_id":row[0],"order_date":row[1],"customer_id":row[2],"customer_name":row[3],
                   "customer_group":row[4],"revenue":0,"status":row[15],"valid_sale":row[16],
                   "amount_due":due_by_order.get(identity),"paid_snapshot":paid_by_order.get(identity),
                   "payment_method":row[17],"channel":row[18],"employee":row[19]})
        if row[1]<target['order_date']:
            target.update(order_date=row[1],customer_id=row[2],customer_name=row[3],customer_group=row[4],
                          amount_due=due_by_order.get(identity),paid_snapshot=paid_by_order.get(identity),
                          payment_method=row[17],channel=row[18],employee=row[19])
        if row[12] is not None: target["revenue"]+=row[12]
    if not sales:
        for name in header_tables:
            m=mappings[name]
            for row in tables[name]:
                oid=val(row,m,"order_id"); day=parse_date(val(row,m,"date"),row.get("_source_year"))
                if oid is None or not day: continue
                cid=val(row,m,"customer_id"); cname=val(row,m,"customer_name")
                st=status(val(row,m,"status"))
                order_rows[(str(oid),day)]={"order_id":str(oid),"order_date":day,"customer_id":customer_key(cid,cname),
                    "customer_name":cname,"customer_group":None,"revenue":parse_number(val(row,m,"revenue")),
                    "status":st,"valid_sale":int(st in {"completed","shipping"}),"amount_due":parse_number(val(row,m,"due")),
                    "paid_snapshot":parse_number(val(row,m,"paid")),"payment_method":val(row,m,"payment_method"),
                    "channel":val(row,m,"channel"),"employee":val(row,m,"employee")}
    order_cols=["order_id","order_date","customer_id","customer_name","customer_group","revenue","status","valid_sale",
                "amount_due","paid_snapshot","payment_method","channel","employee"]
    con.execute(f"CREATE TABLE orders ({','.join(quote(c) for c in order_cols)})")
    con.executemany(f"INSERT INTO orders VALUES ({','.join('?' for _ in order_cols)})",([r[c] for c in order_cols] for r in order_rows.values()))
    con.execute("CREATE INDEX orders_date ON orders(order_date)")
    payment_rows=[]
    for name,m in mappings.items():
        if not re.search(r"soquy|so.quy|thanh.toan|phieu.thu|phieuthu|payment",sources(name)): continue
        if not m["payment_amount"] or not m["payment_date"] or not m["order_id"]: continue
        for row in tables[name]:
            oid=val(row,m,"order_id"); date=parse_date(val(row,m,"payment_date"),row.get("_source_year")); amount=parse_number(val(row,m,"payment_amount"))
            if oid is not None and date and amount is not None: payment_rows.append((str(oid),date,amount))
    con.execute("CREATE TABLE payments(order_id,payment_date,amount)")
    con.executemany("INSERT INTO payments VALUES (?,?,?)",payment_rows)
    con.execute("CREATE INDEX payments_order ON payments(order_id,payment_date)")
    con.execute('CREATE TABLE product_attributes(product_id,prescription)')
    attributes={}
    for name,m in mappings.items():
        if not m['product_id'] or not m['prescription']:continue
        for row in tables[name]:
            pid=val(row,m,'product_id'); flag=val(row,m,'prescription')
            if pid is not None: attributes[str(pid)]=int(fold(flag).strip() in {'1','true','yes','co','dung'})
    con.executemany('INSERT INTO product_attributes VALUES (?,?)',attributes.items())
    con.execute('CREATE TABLE inventory(product_id,product_name,stock_quantity,base_unit)')
    inventory=[]
    for name,m in mappings.items():
        if not m['stock'] or m['date'] or not re.search(r'ton.?kho|inventory',sources(name)):continue
        for row in tables[name]:
            pid=val(row,m,'product_id');title=val(row,m,'product_name')
            pr,pm=product_index.get(('id',str(pid)),(None,None)) if pid else (None,None)
            if pr:title=val(pr,pm,'product_name') or title
            if title:inventory.append((pid,title,parse_number(val(row,m,'stock')),val(row,m,'base_unit')))
    con.executemany('INSERT INTO inventory VALUES (?,?,?,?)',inventory)
    conversion_rows=[(title,large,factor,small) for (title,large),(factor,small) in unit_index.items()]
    con.execute('CREATE TABLE unit_conversions(product_key,large_unit,factor,base_unit)')
    con.executemany('INSERT INTO unit_conversions VALUES (?,?,?,?)',conversion_rows)
    for name,cols in [("sales",columns),("orders",order_cols),("payments",["order_id","payment_date","amount"])]:
        count=con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        info["tables"][name]={"columns":cols,"rows":count,"sources":[{"file":"derived","sheet":"mapped from typed source columns"}],
                              "original_headers":cols,"sample":con.execute(f"SELECT * FROM {name} LIMIT 2").fetchall()}
    dates=[r[1] for _,r in sales] or [r["order_date"] for r in order_rows.values()]
    info["sales_layer"]={"line_tables":line_tables,"header_tables":header_tables,"mappings":mappings,
                         "date_min":min(dates) if dates else None,"date_max":max(dates) if dates else None,
                         "can_count_orders":bool(order_rows) and all(r[0] is not None for _,r in sales),"has_sales_lines":bool(sales),"has_payment_events":bool(payment_rows),
                         "unknown_status_lines":sum(r[15]=="unknown" for _,r in sales) or sum(r['status']=='unknown' for r in order_rows.values()),
                         "missing_amount_rows":sum(r[12] is None for _,r in sales),
                         "missing_quantity_rows":sum(r[9] is None for _,r in sales)}
    info["sales_layer"]["unresolved_date_rows"]=dropped_dates
    cutoff=re.search(r'du lieu co den het\s+(\d{1,2}/\d{1,2}/20\d{2})',fold(readme))
    info['sales_layer'].update(has_prescription_attributes=bool(attributes),has_inventory_snapshot=bool(inventory),
                               inventory_as_of=iso_date(cutoff[1]) if cutoff else None,
                               has_purchase_data=any(re.search(r'nhap|purchase',sources(name)) for name in mappings))
    for table in ['inventory','unit_conversions','product_attributes']:
        cols=[r[1] for r in con.execute('PRAGMA table_info('+table+')')]
        info['tables'][table]={'columns':cols,'rows':con.execute('SELECT COUNT(*) FROM '+table).fetchone()[0],
                               'sources':[{'file':'derived','sheet':'typed source mappings'}],'sample':con.execute('SELECT * FROM '+table+' LIMIT 2').fetchall()}

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from datamate_bench.data import ReadOnlySQL, build_database, iso_date, number
from datamate_bench.normalized import build_normalized
from datamate_bench.metrics import execute
from datamate_bench.temporal import context,resolve,event_comparison,month_series


class CalendarTests(unittest.TestCase):
    def test_intent_boundaries_do_not_substitute_metrics(self):
        from datamate_bench.agent import suggested_operation,type_hint
        self.assertEqual(suggested_operation('Đếm số đơn đã hủy tháng 4/2026'),'count_cancelled')
        self.assertEqual(suggested_operation('Bao nhiêu % đơn hủy tháng này'),'cancel_rate')
        self.assertEqual(suggested_operation('Tổng dthu trong ngày hôm nay'),'revenue')
        self.assertIsNone(suggested_operation('Doanh thu sản phẩm Trà tháng này'))
        self.assertIsNone(suggested_operation('Doanh thu Trà tháng này'))
        self.assertEqual(type_hint('Top 3 khách mua nhiều tiền nhất tuần rồi là ai'),'list')
        self.assertIsNone(type_hint('5 khách nào nợ nhiều nhất, mỗi người còn nợ bao nhiu'))

    def test_previous_week_not_last_seven_days(self):
        self.assertEqual(resolve('tuần rồi','2026-03-04'),['2026-02-23','2026-03-01'])
        self.assertEqual(context('2026-03-31')['previous_month_same_days'],['2026-02-01','2026-02-28'])
        self.assertEqual(resolve('tháng 7 năm 2025','2026-02-01'),['2025-07-01','2025-07-31'])
        self.assertEqual(number('1tr440'),1440000)
        self.assertEqual(event_comparison('7 ngày kể từ ngày 17/02/2026 so với 7 ngày ngay trước ngày đó'),
                         (['2026-02-17','2026-02-23'],['2026-02-10','2026-02-16']))
        self.assertEqual(month_series('từ tháng 11/2025 đến tháng 1/2026'),['2025-11','2025-12','2026-01'])

    def test_changed_rulebook_blocks_precompiled_metric(self):
        from datamate_bench.agent import solve
        class Model:
            def __init__(self):self.actions=iter([{'action':'metric','operation':'revenue'},{'action':'finish','type':'abstain'}])
            def chat(self,*args,**kwargs):return json.dumps(next(self.actions)),{}
        result=solve({'question':'Doanh thu tháng 1/2025','as_of_date':'2025-01-31'},'',
                     '## 1. Luật mới\nTính cả đơn hủy.',
                     {'tables':{},'warnings':[],'sales_layer':{'date_min':'2025-01-01'}},None,Model(),max_steps=2)
        self.assertIn('differs from verified metric contract',result['trace'][0]['error'])
        self.assertEqual(result['prediction'],{'type':'abstain'})

    def test_missing_data_claim_needs_source_evidence(self):
        from datamate_bench.agent import solve
        class Model:
            def __init__(self):self.actions=iter([{'action':'finish','type':'unanswerable','reason':'guess'},{'action':'finish','type':'abstain'}])
            def chat(self,*args,**kwargs):return json.dumps(next(self.actions)),{}
        result=solve({'question':'?','as_of_date':'2025-01-01'},'','intro',{'tables':{},'warnings':[]},None,Model(),max_steps=2)
        self.assertIn('before claiming missing data',result['trace'][0]['error'])
        self.assertEqual(result['prediction'],{'type':'abstain'})


class CanonicalTests(unittest.TestCase):
    def test_revenue_growth_uses_same_period_previous_year(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'metrics.sqlite';con=sqlite3.connect(path)
            con.execute('CREATE TABLE orders(order_date,revenue,valid_sale,status)')
            con.executemany('INSERT INTO orders VALUES (?,?,1,\'completed\')',[('2025-01-01',100),('2025-12-01',500),('2026-01-01',110)])
            con.commit();con.close();sql=ReadOnlySQL(path)
            try:
                q={'question':'Doanh thu tháng 1/2026 so cùng kỳ năm trước thay đổi bao nhiêu %','as_of_date':'2026-01-31'}
                layer={'date_min':'2025-01-01','date_max':'2026-01-31','can_count_orders':True}
                value=execute({'operation':'revenue_growth','compare_start':'2025-12-01','compare_end':'2025-12-31'},q,sql,layer)
                self.assertEqual(value['prediction'],{'type':'percent','answer':10.0})
            finally:sql.close()

    def test_negative_return_invalidates_original_order_across_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shop=root/'shop';shop.mkdir()
            calendar=root/'calendar.csv';calendar.write_text('date,day\n2025-01-01,1\n')
            (shop/'sales.csv').write_text('Mã đơn hàng,Ngày,Hàng,Số lượng,Đơn vị tính,Thành tiền,Trạng thái\nA,01/01/2025,Trà,2,chai,1tr440,Xong\nA,02/01/2025,Trà,-1,chai,-720000,Xong\nB,01/01/2025,Trà,1,chai,100,Xong\n',encoding='utf8')
            raw,meta,_=build_database(shop,calendar,root/'cache')
            typed,meta,_=build_normalized(raw,meta,'Dòng số lượng âm là đơn trả hàng.',root/'typed');sql=ReadOnlySQL(typed)
            try:
                q={'question':'tháng 1/2025','as_of_date':'2025-01-31'}
                self.assertEqual(execute({'operation':'revenue'},q,sql,meta['sales_layer'])['prediction']['answer'],100)
                self.assertEqual(sql.query("SELECT COUNT(*) FROM sales WHERE status='returned'")['rows'][0][0],2)
                self.assertEqual(sql.query("SELECT COUNT(*) FROM orders")['rows'][0][0],2)
                self.assertEqual(sql.query("SELECT order_date FROM orders WHERE order_id='A'")['rows'][0][0],'2025-01-01')
            finally:sql.close()

    def test_quantity_suffixes_conversion_and_growth_direction(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shop=root/'shop';shop.mkdir()
            calendar=root/'calendar.csv';calendar.write_text('date,day\n2025-01-01,1\n')
            (shop/'sales.csv').write_text('Mã đơn hàng,Ngày,Tên hàng,Số lượng,Đơn vị tính,Thành tiền,Trạng thái\nA,01/01/2025,Trà,2 thùng,thùng,240,Xong\nB,02/01/2025,Trà,8 chai,chai,80,Xong\n',encoding='utf8')
            (shop/'danh_muc.csv').write_text('Tên hàng,Đơn vị quy đổi,Tỉ lệ quy đổi,Đơn vị cơ bản\nTrà,thùng,12,chai\n',encoding='utf8')
            raw,meta,_=build_database(shop,calendar,root/'cache')
            typed,meta,_=build_normalized(raw,meta,'',root/'typed');sql=ReadOnlySQL(typed)
            try:
                q={'question':'số chai Trà trong 1 ngày kể từ ngày 02/01/2025 so với 1 ngày ngay trước đó tăng giảm bao nhiêu %','as_of_date':'2025-01-02'}
                result=execute({'operation':'quantity_growth','product':'Trà','start':'2025-01-01','end':'2025-01-01','compare_start':'2025-01-02','compare_end':'2025-01-02'},q,sql,meta['sales_layer'])
                self.assertEqual(result['prediction']['answer'],-66.67)
                self.assertEqual(meta['sales_layer']['missing_quantity_rows'],0)
            finally:sql.close()

    def test_compact_joined_exports_serial_dates_and_payment_cutoff(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shop=root/'shop';shop.mkdir()
            calendar=root/'calendar.csv';calendar.write_text('date,day\n2025-01-01,1\n')
            (shop/'don_hang.csv').write_text('MaHoaDon,NgayLap,TinhTrang,MaKhachHang,TongThanhToan,KenhBanHang\nA,45658,Hoàn thành,K1,100,store\nB,45659,Trả hàng,K1,200,zalo_oa\nC,45660,Hủy,K2,300,store\n',encoding='utf8')
            (shop/'chi_tiet_don.csv').write_text('MaHoaDon,MaHangHoa,SL,SLQuyDoi,TriGia,GiaDonVi\nA,P1,1,6,100,60\nB,P1,2,12,200,100\nC,P1,3,18,300,100\n',encoding='utf8')
            (shop/'khach_hang.csv').write_text('MaKhachHang,TenGoi,Hạng,Loại hình khách\nK1,Lan,thân thiết,Cá nhân\nK2,Tú,lẻ,Cá nhân\n',encoding='utf8')
            (shop/'san_pham.csv').write_text('MaHangHoa,TenGoi,DonViCoSo,DanhMuc\nP1,Trà,chai,Đồ uống\n',encoding='utf8')
            (shop/'thanh_toan.csv').write_text('MaHoaDon,NgayGioThu,SoTienThu\nA,45658,40\nA,45660,60\nB,45659,50\n',encoding='utf8')
            (shop/'thuoc.csv').write_text('MaHangHoa,ThuocKeDon\nP1,true\n',encoding='utf8')
            (shop/'TonKho.csv').write_text('Tên hàng,Tồn thực,Đơn vị cơ bản\nTrà,56,chai\n',encoding='utf8')
            (shop/'danh_muc.csv').write_text('Tên hàng,ĐV lớn,ĐV nhỏ,Hệ số QĐ\nTrà,thùng,chai,10\n',encoding='utf8')
            raw,meta,_=build_database(shop,calendar,root/'cache')
            typed,meta,_=build_normalized(raw,meta,'Dữ liệu có đến hết 03/01/2025.',root/'typed');sql=ReadOnlySQL(typed)
            try:
                self.assertEqual(iso_date('45658.75'),'2025-01-01')
                self.assertEqual(meta['sales_layer']['unresolved_date_rows'],0)
                q={'question':'tháng 1/2025','as_of_date':'2025-01-03'}
                layer=meta['sales_layer']
                self.assertEqual(execute({'operation':'cancel_rate'},q,sql,layer)['prediction']['answer'],33.33)
                self.assertEqual(execute({'operation':'quantity','product':'Trà','unit':'base'},q,sql,layer)['prediction']['answer'],6)
                self.assertEqual(execute({'operation':'revenue_by','dimension':'channel','values':['quay','zalo oa']},q,sql,layer)['prediction']['answer'],[100,0])
                self.assertEqual(execute({'operation':'debt','as_of':'2025-01-02','by_customer':False},q,sql,layer)['prediction']['answer'],210)
                self.assertEqual(execute({'operation':'debt_customers','as_of':'2025-01-02'},q,sql,layer)['prediction']['answer'],1)
                self.assertEqual(execute({'operation':'prescription_revenue'},q,sql,layer)['prediction']['answer'],100)
                self.assertEqual(execute({'operation':'count_cancelled'},q,sql,layer)['prediction']['answer'],1)
                self.assertEqual(execute({'operation':'stock_units','product':'Trà','unit':'thùng','as_of':'2025-01-03'},q,sql,layer)['prediction']['answer'],[5,6])
                self.assertEqual(execute({'operation':'customer_group_share','group':'thân thiết'},q,sql,layer)['prediction']['answer'],100)
                self.assertEqual(execute({'operation':'product_group_share','group':'Đồ uống'},q,sql,layer)['prediction']['answer'],100)
                self.assertEqual(execute({'operation':'product_group_revenue','group':'Đồ uống'},q,sql,layer)['prediction']['answer'],100)
                self.assertEqual(execute({'operation':'monthly_totals','months':['2025-01','2025-02']},q,sql,layer)['prediction']['answer'],[100,0])
                self.assertEqual(execute({'operation':'sale_price','product':'Trà','start':'2025-01-01','end':'2025-01-01'},q,sql,layer)['prediction']['answer'],10)
                self.assertEqual(execute({'operation':'purchase_average'},q,sql,layer)['prediction']['type'],'unanswerable')
                with self.assertRaises(ValueError):execute({'operation':'customer_group_share','group':'unknown'},q,sql,layer)
                with self.assertRaises(ValueError): execute({'operation':'revenue','channel':'store'},q,sql,layer)
                with self.assertRaises(ValueError): execute({'operation':'quantity','unit':'banana'},q,sql,layer)
            finally:sql.close()

    def test_money_dates_whole_order_returns_and_metric_types(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); shop=root/'shop'; shop.mkdir()
            calendar=root/'calendar.csv'; calendar.write_text('date,lunar_day\n2025-08-01,8\n')
            (shop/'sales.csv').write_text(
                'Mã đơn hàng,Ngày,Tên hàng,Số lượng,Thành tiền,Trạng thái,Khách hàng\n'
                'A,01/08/2025,Cà phê,1,20.000đ,Xong,Lan\n'
                'A,01/08/2025,Trà,1,10.000đ,,Lan\n'
                'B,01/08/2025,Cà phê,1,20.000đ,Xong,Tú\n'
                'B,01/08/2025,Trà,1,10.000đ,Trả hàng,Tú\n'
                'C,02/08/2025,Cà phê,1,20.000đ,Hủy,An\n'
                'D,02/08/2025,Cà phê,1,40.000đ,Đang giao,An\n',encoding='utf-8')
            with (shop/'sales.csv').open('a',encoding='utf8') as stream:
                stream.write(',,Cộng,,120.000đ,,\n,,Ghi chú: tiền tính bằng đồng,,,,\n')
            raw,meta,_=build_database(shop,calendar,root/'cache')
            typed,meta,_=build_normalized(raw,meta,'',root/'typed')
            sql=ReadOnlySQL(typed)
            q={'question':'Doanh thu tháng 8/2025','as_of_date':'2025-08-31'}
            try:
                result=execute({'operation':'revenue'},q,sql,meta['sales_layer'])
                self.assertEqual(result['prediction'],{'type':'number','answer':70000})
                self.assertEqual(execute({'operation':'count_orders'},q,sql,meta['sales_layer'])['prediction']['answer'],2)
                self.assertEqual(execute({'operation':'best_day'},q,sql,meta['sales_layer'])['prediction']['answer'],'2025-08-02')
                self.assertEqual(sql.query("SELECT thanh_tien FROM raw_t01 LIMIT 1")['rows'][0][0],'20.000đ')
                self.assertEqual(sql.query("SELECT thanh_tien FROM t01 LIMIT 1")['rows'][0][0],20000)
                self.assertEqual(sql.query("SELECT count(*) FROM sales WHERE status='returned'")['rows'][0][0],2)
            finally: sql.close()

    def test_no_order_ids_do_not_invent_order_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); shop=root/'shop'; shop.mkdir()
            calendar=root/'calendar.csv'; calendar.write_text('date,day\n2025-01-01,1\n')
            (shop/'sales.csv').write_text('Ngày,Hàng,Số lượng,Thành tiền,Trạng thái\n01/01/2025,A,1,100,Xong\n,A,1,100,Xong\n',encoding='utf-8')
            raw,meta,_=build_database(shop,calendar,root/'cache')
            typed,meta,_=build_normalized(raw,meta,'',root/'typed')
            sql=ReadOnlySQL(typed)
            try:
                q={'question':'tháng 1/2025','as_of_date':'2025-01-31'}
                self.assertEqual(execute({'operation':'revenue'},q,sql,meta['sales_layer'])['prediction']['answer'],200)
                with self.assertRaises(ValueError): execute({'operation':'count_orders'},q,sql,meta['sales_layer'])
            finally: sql.close()

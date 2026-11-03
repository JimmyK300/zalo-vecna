import sqlite3
import tempfile
import unittest
from pathlib import Path

from datamate_bench.agent import suggested_operation, type_hint
from datamate_bench.data import build_database, ReadOnlySQL
from datamate_bench.metrics import execute
from datamate_bench.normalized import build_normalized
from datamate_bench.temporal import resolve,event_comparison


class SQLImprovements(unittest.TestCase):
    def test_inline_purchase_suppliers_survive_receipt_aggregation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shop=root/'shop';shop.mkdir()
            calendar=root/'calendar.csv';calendar.write_text('date,day\n2025-01-01,1\n')
            (shop/'sales.csv').write_text('Mã đơn hàng,Ngày,Tên hàng,Số lượng,Thành tiền,Trạng thái\nA,01/01/2025,Trà,1,100,Xong\n',encoding='utf-8')
            (shop/'so_nhap.csv').write_text('Số phiếu,Ngày nhập,Nhà cung cấp,Tên hàng,SL nhập,Giá nhập,Thành tiền\nP1,01/01/2025,Cty A,Trà,2,100,200\nP1,01/01/2025,Cty A,Cà phê,3,200,600\nP2,02/01/2025,Cty B,Trà,1,100,100\n',encoding='utf-8')
            raw,meta,_=build_database(shop,calendar,root/'cache')
            typed,meta,_=build_normalized(raw,meta,'',root/'typed');sql=ReadOnlySQL(typed)
            try:
                self.assertEqual(sql.query('SELECT supplier,SUM(amount) FROM purchase_receipts GROUP BY supplier ORDER BY SUM(amount) DESC')['rows'],[('Cty A',800),('Cty B',100)])
                self.assertEqual(meta['sales_layer']['purchase_warnings'],[])
            finally:sql.close()

    def test_semantic_boundaries(self):
        self.assertIsNone(suggested_operation('Doanh thu các đơn chuyển khoản tháng này là bao nhiêu?'))
        self.assertEqual(type_hint('Giá Otrivin 0,1% xịt mũi chai 10ml ngày 25/07/2025 là bao nhiêu?'),None)
        self.assertEqual(type_hint('Thuốc Otrivin 0,1% chiếm bao nhiêu % doanh thu?'),'percent')
        self.assertEqual(resolve('Từ 1/1 năm nay đến hết hnay bao nhiêu đơn?','2026-09-13'),['2026-01-01','2026-09-13'])
        self.assertEqual(resolve('Từ ngày 1/1 đến hết hnay','2026-09-13'),['2026-01-01','2026-09-13'])
        self.assertEqual(resolve('Tổng doanh thu quý 3/2025','2026-01-01'),['2025-07-01','2025-09-30'])
        self.assertEqual(event_comparison('Ngày 15/02/2025 tăng giá, 3 ngày kể từ ngày đó so với 3 ngày ngay trước đó'),
                         (['2025-02-15','2025-02-17'],['2025-02-12','2025-02-14']))

    def test_purchase_receipt_totals_weighted_prices_and_catalog_priority(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shop=root/'shop';shop.mkdir()
            calendar=root/'calendar.csv';calendar.write_text('date,day\n2025-01-01,1\n')
            files={
                'sales.csv':'Mã đơn hàng,Ngày,Tên hàng,Số lượng,Đơn vị tính,Thành tiền,Trạng thái,Phương thức thanh toán\nA,01/01/2025,Thuốc A,1,hộp,100,Xong,Tiền mặt\nB,02/01/2025,Thuốc B,1,hộp,300,Xong,Chuyển khoản\n',
                'a_danh_quy_doi.csv':'Tên hàng,Đơn vị quy đổi,Tỉ lệ quy đổi,Đơn vị cơ bản\nThuốc A,thùng,10,hộp\n',
                'z_san_pham.csv':'Mã hàng,Tên hàng,Đơn vị cơ bản,Nhóm hàng\nP1,Thuốc A,hộp,Nhóm thuốc\nP2,Thuốc B,hộp,Nhóm thuốc\n',
                'thuoc.csv':'Mã hàng,Thuốc kê đơn\nP1,true\nP2,false\n',
                'nhap_header.csv':'Mã nhập hàng,Thời gian,Cần trả NCC,Trạng thái\nR1,01/01/2025,200,Đã nhập hàng\nR2,02/01/2025,900,Đã nhập hàng\nR3,03/01/2025,999,Hủy\n',
                'chi_tiet_nhap.csv':'Mã nhập hàng,STT,Mã hàng,ĐVT,Số lượng,Đơn giá\nR1,1,P1,thùng,2,100\nR2,1,P1,thùng,3,200\nR2,2,P2,hộp,1,300\nR3,1,P1,thùng,1,999\n',
            }
            for name,text in files.items():(shop/name).write_text(text,encoding='utf-8')
            raw,meta,_=build_database(shop,calendar,root/'cache')
            typed,meta,_=build_normalized(raw,meta,'',root/'typed');sql=ReadOnlySQL(typed)
            q={'question':'tháng 1/2025','as_of_date':'2025-01-31'};layer=meta['sales_layer']
            try:
                # Expected independently: (2*100 + 3*200)/(2+3) = 160, not AVG(100,200).
                self.assertEqual(execute({'operation':'purchase_average','product':'Thuốc A','unit':'thùng'},q,sql,layer)['prediction']['answer'],160)
                self.assertEqual(execute({'operation':'purchase_average','product':'Thuốc A','unit':'base_unit'},q,sql,layer)['prediction']['answer'],16)
                # Header R2 spans two details; count its total only once; exclude R3.
                self.assertEqual(execute({'operation':'purchase_total'},q,sql,layer)['prediction']['answer'],1100)
                self.assertEqual(execute({'operation':'prescription_share'},q,sql,layer)['prediction']['answer'],25)
                self.assertEqual(execute({'operation':'revenue_by','dimension':'payment_method','values':['Tiền mặt','cash','transfer']},q,sql,layer)['prediction']['answer'],[100,100,300])
                self.assertEqual(execute({'operation':'revenue_share','dimension':'payment_method','values':['cash']},q,sql,layer)['prediction']['answer'],25)
                self.assertEqual(sql.query("SELECT product_id FROM sales WHERE product_name='Thuốc A'")['rows'][0][0],'P1')
            finally:sql.close()

    def test_explicit_growth_year_overrides_both_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'facts.sqlite';con=sqlite3.connect(path)
            con.execute('CREATE TABLE orders(order_date,revenue,valid_sale,status)')
            con.executemany('INSERT INTO orders VALUES (?,?,1,\'completed\')',[('2025-04-01',100),('2025-05-01',125),('2026-04-01',500)])
            con.commit();con.close();sql=ReadOnlySQL(path)
            try:
                result=execute({'operation':'revenue_growth','start':'2026-05-01','end':'2026-05-31','compare_start':'2026-04-01','compare_end':'2026-04-30'},
                               {'question':'Doanh thu tháng 5/2025 so với tháng trước tăng bao nhiêu phần trăm?','as_of_date':'2026-05-31'},sql,
                               {'date_min':'2025-01-01','date_max':'2026-05-31','can_count_orders':True})
                self.assertEqual(result['prediction']['answer'],25)
            finally:sql.close()


if __name__=='__main__':unittest.main()

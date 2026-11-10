"""Compose train-only DataMate questions and compute labels with read-only SQL.

Run from the repository root:
    python -m scripts.compose_queries --data train --businesses-from ../zalo-ft1-data/sft_nothink_opt1.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict, deque
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from datamate.io_contract import find_dataset_dir, load_questions, load_rulebook
from datamate.primitives import period
from datamate.rules import active_rules
from datamate.semantics import strip_accents
from datamate.workspace import load_workspace


LEVELS = ("easy", "medium", "hard", "expert")
REVENUE = "SELECT SUM(revenue_lines) FROM orders WHERE in_revenue AND order_date BETWEEN ? AND ?"
ORDER_COUNT = "SELECT COUNT(*) FROM orders WHERE in_revenue AND order_date BETWEEN ? AND ?"
CANCEL_RATE = """SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'huy') / NULLIF(COUNT(*), 0), 2)
FROM orders WHERE order_date BETWEEN ? AND ?"""
TOP_PRODUCTS = """SELECT MIN(item) AS item, SUM(qty_base) AS qty, item_key FROM sale_lines
WHERE order_date BETWEEN ? AND ? GROUP BY item_key ORDER BY qty DESC, item_key ASC LIMIT ?"""
PRODUCT_VOLUME = """SELECT COALESCE(SUM(qty_base), 0) FROM sale_lines
WHERE order_date BETWEEN ? AND ? AND item_key = ?"""
BUYER_SHARE = """SELECT 100.0 * COUNT(*) FILTER (WHERE n >= ?) / NULLIF(COUNT(*), 0)
FROM (
    SELECT customer_code, COUNT(*) AS n FROM orders
    WHERE in_revenue AND NOT is_walkin AND order_date BETWEEN ? AND ?
    GROUP BY customer_code
)"""


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def wording(text):
    return " ".join(re.findall(r"[a-z0-9]+", strip_accents(text)))


def load_primitives(root):
    spec = load_json(root / "query/query_rulebook.json")
    names = {f["name"] for section in spec.values() for f in section.get("functions", [])
             if f.get("scope", "train") == "train"}
    templates = load_json(root / "query/query_templates.json")
    examples = {e["id"]: e for e in templates["examples"]}
    rule_ids = {}
    for entry in load_json(root / "rulebooks/primitives_library.json"):
        origins = [entry["rulebook_origin"], *entry.get("other_origins", [])]
        if any(origin["source_file"] == "train/rulebook.md" for origin in origins):
            rule_ids[entry["name"]] = entry["primitive_id"]
    return names, examples, rule_ids


def prepared_tables(db, ws):
    table = ws.tables.get("orders")
    if table is None or any(name.startswith("orders_") for name in ws.tables):
        return None
    orders = table.df
    if not {"order_id", "order_date", "in_revenue", "status"} <= set(orders):
        return None
    if orders.empty or orders[["order_date", "status", "in_revenue"]].isna().any().any():
        return None
    if orders.order_id.isna().any() or orders.order_id.duplicated().any():
        return None
    if not set(orders.status) <= {"hoan_thanh", "huy", "tra_hang", "dang_giao"}:
        return None
    db.register("order_frame", orders)
    db.execute("CREATE TEMP TABLE orders AS SELECT * FROM order_frame")
    revenue = "revenue_lines" in orders and orders.loc[orders.in_revenue, "revenue_lines"].notna().all()
    buyers = {"customer_code", "is_walkin"} <= set(orders) and orders.is_walkin.notna().all()
    if buyers:
        codes = orders.loc[orders.in_revenue & ~orders.is_walkin.astype(bool), "customer_code"]
        buyers = not codes.empty and codes.notna().all() and codes.astype(str).str.strip().ne("").all()
    lines = ws.lines
    products = False
    if lines is not None:
        lines = lines.loc[lines.in_revenue & ~lines.is_fee].copy()
        if not lines.empty and lines[["item", "qty_base"]].notna().all().all():
            # shortcut: use a name only when the shop has no item codes, pending a reliable mixed-key mapping.
            codes = lines.item_code.dropna()
            if codes.empty or len(codes) == len(lines):
                lines["item"] = lines.item.astype(str).str.strip()
                key = lines.item if codes.empty else lines.item_code
                lines["item_key"] = key.astype(str).str.strip()
                products = lines.item.ne("").all() and lines.item_key.ne("").all() and lines.qty_base.ge(0).all()
                if products:
                    db.register("line_frame", lines[["order_date", "item", "item_key", "qty_base"]])
                    db.execute("CREATE TEMP TABLE sale_lines AS SELECT * FROM line_frame")
    return orders.order_date.min().date(), orders.order_date.max().date(), revenue, buyers, products


def compose(db, flags, anchors, examples):
    lo, _, revenue, buyers, products = flags
    for anchor in anchors:
        day = (anchor, anchor)
        days = (7, 14, 30)[anchor.toordinal() % 3]
        recent = period("last_n_days", anchor, n=days)
        current = period("this_month", anchor)
        previous = period("last_month_same_days", anchor)
        if recent[0] < lo or previous[0] < lo:
            continue
        yield "easy", "daily_orders", "number", ORDER_COUNT, day, (
            f"Ngày {anchor:%d/%m/%Y} cửa hàng có bao nhiêu đơn?", "today", "order_count"
        ), ["Số đơn"], None
        if revenue:
            yield "easy", "recent_revenue", "number", REVENUE, recent, (
                f"Tính tới {anchor:%d/%m/%Y}, {days} ngày gần nhất cửa hàng bán được bao nhiêu tiền?",
                "last_n_days", "order_revenue"
            ), ["Bán được bao nhiêu tiền, thu về bao nhiêu tiền"], None
            now = db.execute(REVENUE, current).fetchone()[0]
            before = db.execute(REVENUE, previous).fetchone()[0]
            if now is not None and before:
                yield "hard", "month_to_date_growth", "percent", None, [current, previous], (
                    f"Mốc hôm nay là {anchor:%d/%m/%Y}. {examples['month_to_date_growth']['question_template']}",
                    "this_month", "last_month_same_days", "order_revenue", "pct_growth"
                ), ["Tháng này", "Phần trăm thay đổi, tăng trưởng"], round(100.0 * (now - before) / before, 2)
        yield "medium", "recent_cancel_rate", "percent", CANCEL_RATE, recent, (
            f"Trong {days} ngày gần nhất tính tới {anchor:%d/%m/%Y}, bao nhiêu phần trăm đơn đặt bị huỷ?",
            "last_n_days", "cancellation_rate"
        ), ["Tỉ lệ đơn huỷ trong kỳ"], None
        if products:
            n = (2, 3, 5)[anchor.toordinal() % 3]
            top = db.execute(TOP_PRODUCTS, [*current, n]).fetchall()
            if len(top) == n and top[-1][1] > 0:
                yield "medium", "top_products", "list", None, [current, n], (
                    f"Từ đầu tháng đến hết {anchor:%d/%m/%Y}, {n} mặt hàng nào bán chạy nhất? "
                    "Liệt kê theo thứ tự xếp hạng.",
                    "this_month", "best_selling_items", "result_order"
                ), ["Tháng này", "Bán chạy"], [r[0] for r in top]
                product, quantity, product_key = top[0]
                product_label = f"mã {product_key}" if product != product_key else f"tên \"{product}\""
                yield "medium", "named_product_quantity", "number", None, [current, product_key], (
                    f"Từ đầu tháng đến hết {anchor:%d/%m/%Y}, mặt hàng có {product_label} bán được "
                    "bao nhiêu đơn vị cơ bản?",
                    "this_month", "valid_revenue_orders", "unit_conversion"
                ), ["Tháng này", "Bán chạy"], round(float(quantity), 10)
                prior_quantity = db.execute(PRODUCT_VOLUME, [*previous, product_key]).fetchone()[0]
                yield "expert", "leader_quantity_delta", "number", None, [current, previous, product_key], (
                    f"Lấy mặt hàng bán chạy nhất từ đầu tháng tới hôm nay ({anchor:%d/%m/%Y}). "
                    "Số lượng quy về đơn vị cơ bản "
                    "của chính mặt hàng đó kỳ này trừ cùng kỳ tháng trước bằng bao nhiêu?",
                    "this_month", "last_month_same_days", "best_selling_items", "result_order", "abs_diff"
                ), ["Tháng này", "Bán chạy"], round(float(quantity - prior_quantity), 10)
        if buyers:
            threshold = (2, 3, 4)[anchor.toordinal() % 3]
            now = db.execute(BUYER_SHARE, [threshold, *current]).fetchone()[0]
            before = db.execute(BUYER_SHARE, [threshold, *previous]).fetchone()[0]
            if now is not None:
                yield "hard", "repeat_buyer_share", "percent", None, [current, threshold], (
                    f"Từ đầu tháng tới hết {anchor:%d/%m/%Y}, khách mua từ {threshold} đơn trở lên "
                    "chiếm bao nhiêu phần trăm số khách đã mua?",
                    "this_month", "customer_identity", "order_count", "distinct_buying_customers"
                ), ["Số đơn", "Xác định khách"], round(now, 2)
            if now is not None and before is not None:
                yield "expert", "repeat_buyer_share_delta", "number", None, [current, previous, threshold], (
                    f"Trong mỗi kỳ, tính phần trăm khách mua từ {threshold} đơn trở lên trong số khách đã mua. "
                    f"Lấy tỉ lệ từ đầu tháng tới hết {anchor:%d/%m/%Y} trừ tỉ lệ cùng kỳ tháng trước. "
                    "Chênh lệch là bao nhiêu "
                    "điểm phần trăm, làm tròn kết quả cuối cùng đến 2 chữ số thập phân?",
                    "this_month", "last_month_same_days", "customer_identity", "order_count",
                    "distinct_buying_customers", "abs_diff"
                ), ["Tháng này", "Số đơn", "Xác định khách"], round(now - before, 2)


def choose(groups, limit, rng):
    """Take turns by (shop, recipe) so one large shop cannot consume a level."""
    queues = []
    for group in groups.values():
        rng.shuffle(group)
        queues.append(deque(group))
    rng.shuffle(queues)
    active = deque(queues)
    selected = []
    while active and len(selected) < limit:
        group = active.popleft()
        selected.append(group.popleft())
        if group:
            active.append(group)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--businesses-from", type=Path, required=True,
                        help="JSONL or JSONL.gz of FT1 train-only rows, used as a shop allowlist")
    parser.add_argument("--primitives", type=Path, default=Path("primitives"))
    parser.add_argument("--out", type=Path, default=Path("out/query_compositions/train.jsonl"))
    parser.add_argument("--per-level", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    if args.per_level < 1 or args.out.exists():
        parser.error("--per-level must be positive and --out must be a new file")
    rulebook = load_rulebook(args.data)
    if not rulebook or "bộ train" not in rulebook.splitlines()[0] or not (args.data / "answers.jsonl").is_file():
        parser.error("--data must point to the released train split")
    rules = active_rules(rulebook)
    required = {"in_revenue", "dup_line", "order_fill", "walkin", "service_fee_not_item",
                "best_seller_base_units", "month_to_date_same_days"}
    if not required <= rules:
        parser.error(f"Missing rule guards: {sorted(required - rules)}")
    questions = load_questions(args.data / "questions.jsonl")
    allowed = set()
    for row in read_rows(args.businesses_from):
        if row["split"] != "train":
            parser.error("--businesses-from must contain train rows only")
        allowed.add(row["business_id"])
    eval_ids = set(load_json(Path("finetune/eval80.json"))["ids"])
    eval_shops = {q["business_id"] for q in questions if q["id"] in eval_ids}
    shops = sorted(({q["business_id"] for q in questions} & allowed) - eval_shops)
    if not shops:
        parser.error("No eligible train shops after excluding eval80 shops")
    seen_explicit = {(q["business_id"], wording(q["question"])) for q in questions}
    refs, examples, rule_ids = load_primitives(args.primitives)
    shared = {"calendar": pd.read_csv(args.data / "calendar.csv")}
    pools = defaultdict(lambda: defaultdict(list))
    rng = random.Random(args.seed)
    for shop in shops:
        ws = load_workspace(find_dataset_dir(args.data, shop), shared, rules)
        with duckdb.connect(":memory:", config={"threads": 1}) as db:
            flags = prepared_tables(db, ws)
            if flags is None:
                print(f"{shop}: skipped, no single complete order table", flush=True)
                continue
            lo, hi, *_ = flags
            anchors = [lo + timedelta(days=d) for d in range((hi - lo).days + 1)]
            before = sum(len(pools[level][key]) for level in LEVELS for key in pools[level])
            for anchor in anchors:
                for level, recipe, typ, sql, params, text, source, answer in compose(db, flags, [anchor], examples):
                    question, *primitives = text
                    assert set(primitives) <= refs, primitives
                    anchor_str = anchor.isoformat()
                    normalized = wording(question)
                    if (shop, normalized) in seen_explicit:
                        continue
                    if sql:
                        answer = db.execute(sql, params).fetchone()[0]
                    if answer is None:
                        continue
                    if recipe == "recent_revenue":
                        answer = int(round(answer))
                    key = (shop, anchor_str, recipe)
                    qid = "syn_" + hashlib.sha256("/".join(key).encode()).hexdigest()[:20]
                    pools[level][(shop, recipe)].append({
                        "id": qid, "business_id": shop, "as_of_date": anchor_str, "question": question,
                        "type": typ, "answer": answer, "difficulty": level, "recipe": recipe,
                        "primitive_refs": primitives, "rulebook_ids": [rule_ids[name] for name in source],
                        "parameters": params,
                    })
            after = sum(len(pools[level][key]) for level in LEVELS for key in pools[level])
            print(f"{shop}: {after - before} candidates", flush=True)
    selected = [row for level in LEVELS for row in choose(pools[level], args.per_level, rng)]
    counts = Counter(row["difficulty"] for row in selected)
    if len(counts) != len(LEVELS):
        raise SystemExit(f"No output written. Missing difficulty levels: {dict(counts)}")
    lines = [json.dumps(row, ensure_ascii=False, allow_nan=False, default=date.isoformat) for row in selected]
    assert len({row["id"] for row in selected}) == len(selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as out:
        out.write("\n".join(lines) + "\n")
    print(f"{args.out}: {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()

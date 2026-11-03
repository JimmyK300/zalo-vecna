"""Rulebook guard: helpers that encode a business rule only switch on when the rulebook says so.

Code must not silently carry yesterday's rules. Each rule-encoding helper names the rulebook
wording it relies on (accent-insensitive regexes, all must match). If a new rulebook drops or
rewords the rule, the helper is disabled and the profile tells the model to work from the
rulebook text instead. Pure data facts (parsing, roles, joins) need no entry here.
"""
from __future__ import annotations

import re

from .semantics import strip_accents

# helper -> regexes that must all be found in the (accent-stripped, lower-cased) rulebook
RULES: dict[str, tuple[str, ...]] = {
    # §1: returns, even partial, exclude the whole order; in-delivery orders still count
    "in_revenue": (r"khong huy.{0,20}khong tra hang", r"ke ca chi tra mot phan", r"dang giao van tinh"),
    # §3: walk-in orders are not customers
    "walkin": (r"khach le.{0,40}khong phai khach",),
    # §8: identical lines within one order are an export error, count once
    "dup_line": (r"cung mot don.{0,40}giong nhau.{0,80}tinh mot lan",),
    # §8: blank cells on continuation lines repeat the first line of the order
    "order_fill": (r"o de trong.{0,80}giong o do o dong dau",),
    # §4: debt = order total minus payments up to day X, orders not cancelled (returns still count),
    #     sum of orders still owing; COD and marketplace orders are not customer debt
    "debt_ledger": (r"no mot don tai ngay x.{0,40}chua tra cua tong don", r"don cod.{0,60}khong phai khach no",
                    r"don duoc tinh vao no.{0,10}don khong huy"),
    # §4: new debt / repayments in a period only concern "ghi nợ" orders; same-day payment = paid at purchase
    "debt_movements": (r"no them.{0,80}chua tra ngay trong ngay mua", r"tien tra cung ngay mua la tra luc mua"),
    # §3: the service-fee line is not an item (not ranked, not counted), but stays in revenue
    "service_fee_not_item": (r"phi phuc vu.{0,30}khong phai mat hang",),
    # §3/§5: best seller by base-unit quantity; most profitable by gross margin = revenue - qty x cost on the line
    "best_seller_base_units": (r"ban chay.{0,30}so luong quy ve don vi co ban",),
    "gross_margin_line_cost": (r"lai gop.{0,60}doanh thu.{0,40}tru gia von", r"gia von ghi tren dong ban"),
    # (public rulebook) a book without a status column keeps the status in the note; no status note = completed
    "no_status_note": (r"so khong co cot trang thai.{0,40}ghi chu", r"ghi chu khong nhac trang thai.{0,20}hoan thanh"),
    # §2: weeks start on Monday; "tuần này" = Monday..today; "tuần trước" = full previous week
    "week_monday": (r"tuan.{0,10}bat dau.{0,6}thu hai",),
    # §2: "tháng này" compared with last month uses the same number of days
    "month_to_date_same_days": (r"thang nay.{0,200}cung so ngay",),
    # §2: month/quarter without a year = this year unless that is after today, then last year
    "month_no_year": (r"khong ghi nam.{0,80}nam nay.{0,80}nam truoc",),
    # §2: shifts by order creation time; afternoon vs morning compares against the morning
    "shift_hours": (r"ca sang 06:00.{1,3}13:59", r"ca chieu 14:00.{1,3}21:59", r"moc la ca sang"),
    # §3: an item group is the group column, exactly the named group
    "group_exact": (r"nhom hang.{0,10}theo cot nhom hang.{0,30}dung ten nhom",),
    # §6 (coffee): size M / L = item name ends with (M) / (L), case-insensitive
    "size_suffix": (r"size m, size l.{0,30}ten ket thuc bang.{0,6}\(m\).{0,8}\(l\)",),
    # §3: best salesperson by revenue, not order count
    "staff_by_revenue": (r"nhan vien ban gioi nhat.{0,10}theo doanh thu",),
}


def active_rules(rulebook: str | None) -> set[str]:
    """Helpers whose rule wording is present. No rulebook given -> all on (backwards compatible)."""
    if not rulebook:
        return set(RULES)
    text = re.sub(r"[*`_]", "", strip_accents(rulebook))
    text = re.sub(r"\s+", " ", text)
    return {name for name, pats in RULES.items() if all(re.search(p, text) for p in pats)}

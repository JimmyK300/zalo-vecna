"""Tools the agent can call. The LLM decides *how* to compute; these compute."""
from __future__ import annotations

import functools
import io
import threading
import re
import traceback

import numpy as np
import pandas as pd

from .primitives import as_of_balance, pct_change, period, rank
from .workspace import Workspace

MAX_OUTPUT_CHARS = 6000
SQL_TIMEOUT_S = 60

TOOL_SPECS = [
    {
        "name": "run_sql",
        "description": "Run a DuckDB SQL query over the business tables and get the result (first 50 rows). "
                       "Use for filtering, joins and aggregations.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "DuckDB SQL"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_python",
        "description": "Run Python (pandas/numpy). Preloaded: pd, np, tables (dict name->DataFrame copy, incl. "
                       "the derived 'orders' table), con (DuckDB; con.sql(q).df()), docs, and helpers: "
                       "period(kind, as_of, n=None, month=None, quarter=None, year=None) -> (start, end) inclusive, "
                       "kinds today/yesterday/this_week/last_week/this_month/last_month_same_days/last_month/"
                       "year_to_date/last_n_days/month/quarter/this_quarter/year/until/tet(cal=tables['calendar']); "
                       "as_of_balance(plus, key, amount, date, as_of, minus=None, minus_key=None, minus_amount=None, "
                       "minus_date=None) -> per-key plus/minus/balance at the end of as_of (debt: plus=qualifying "
                       "orders with order total, minus=payments, key=order id; stock: receipts minus sales); "
                       "pct_change(curr, prev); rank(df, value, n=None, code=None, name=None) with the rulebook "
                       "tie-break; debt (customer-debt ledger, may be None): debt.debt_table(day, n), debt.debt_total(day, "
                       "older_than_days=None), debt.debt_movements(customer_or_None, start, end), debt.debt_orders(day). "
                       "M (order metrics, may be None): M.revenue((start, end)), M.order_count, M.avg_order, M.cancel_rate, "
                       "M.pct_change(r, r2), M.peak_day(r), M.peak_month([r...]), M.trend([r...]), M.top_customers(r, by, n). "
                       "I (item recipes on `lines`, may be None): I.best_sellers(r, n), I.item_revenue(name, r), I.item_units(name, r), "
                       "I.group_share(group, r), I.most_profitable(r, n). "
                       "Variables persist between calls. Print what you need to see.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "final_answer",
        "description": "Submit the final result. Call exactly once when done.",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {"type": "string",
                         "enum": ["number", "percent", "date", "boolean", "string", "list", "table",
                                  "unanswerable", "abstain"]},
                "answer": {"description": "The value in the exact format of the type (see rulebook §9). "
                                          "Null for unanswerable/abstain."},
                "confidence": {"type": "number", "description": "0..1 probability the answer is correct"},
                "reason": {"type": "string", "description": "One-sentence justification (Vietnamese ok)"},
            },
            "required": ["type", "answer", "confidence", "reason"],
            "additionalProperties": False,
        },
    },
]


def _metrics(ws):
    o = ws.tables.get("orders")
    if o is None or not {"order_date", "in_revenue", "revenue_lines", "status"} <= set(o.df.columns):
        return None
    from .metrics import Metrics
    return Metrics(o.df)


def _items(ws):
    if getattr(ws, "lines", None) is None:
        return None
    from .items import Items
    return Items(ws.lines, ws.rules)


def _truncate(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT_CHARS else s[:MAX_OUTPUT_CHARS] + f"\n...[truncated {len(s) - MAX_OUTPUT_CHARS} chars]"


class ToolExecutor:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.con = ws.connect()
        self.ns = {
            "pd": pd, "np": np, "con": self.con, "docs": dict(ws.docs),
            "tables": {t.name: t.df.copy() for t in ws.tables.values()},
            "rank": rank, "period": period, "as_of_balance": as_of_balance, "pct_change": pct_change,
            "debt": getattr(ws, "debt", None),
            "M": _metrics(ws),
            "I": _items(ws),
        }
        self.outputs: list[str] = []  # everything the model has seen, for evidence checks
        if getattr(ws, "cleaner", False):
            self.ns["raw_tables"] = {name: df.copy(deep=True) for name, df in ws.raw_tables.items()}
            self.ns["docs"]["cleaning_report"] = str(ws.cleaning_reports)

    def run(self, name: str, args: dict) -> str:
        try:
            if name == "run_sql":
                out = self._sql(args["query"])
            elif name == "run_python":
                out = self._python(args["code"])
            else:
                out = f"Unknown tool {name}"
        except Exception as e:  # noqa: BLE001
            out = f"ERROR {type(e).__name__}: {e}" + self._hint(name, args, e)
        out = _truncate(out)
        self.outputs.append(out)
        return out

    def _hint(self, name: str, args: dict, e: Exception) -> str:
        """Small models repeat the same broken query; tell them how to fix it."""
        msg = str(e)
        hints = []
        if name == "run_sql":
            if "column" in msg.lower() and ("not found" in msg.lower() or "does not have" in msg.lower()):
                q = args.get("query", "")
                for t in self.ws.tables.values():
                    if re.search(rf"\b{re.escape(t.name)}\b", q):
                        hints.append(f"columns of {t.name}: {', '.join(map(str, t.df.columns))}")
            if "date_format" in msg:
                hints.append("DuckDB uses strftime(col, '%Y-%m') or date_trunc('month', col)")
            if re.search(r"FLOAT64|INT64|STRING|DATETIME", msg):
                hints.append("DuckDB types: DOUBLE, BIGINT, VARCHAR, DATE, TIMESTAMP")
            if "VARCHAR" in msg and ("compare" in msg or "convert" in msg.lower()):
                hints.append("a text column: use the parsed '__date' column, or TRY_CAST(col AS DOUBLE)")
            if "Parser Error" in msg:
                hints.append("one statement per call; DuckDB SQL dialect; column names are plain snake_case")
        return ("\nHINT: " + " | ".join(hints)) if hints else ""

    def _sql(self, query: str) -> str:
        timer = threading.Timer(SQL_TIMEOUT_S, self.con.interrupt)
        timer.start()
        try:
            df = self.con.sql(query).df()
        finally:
            timer.cancel()
        with pd.option_context("display.max_columns", 50, "display.width", 250, "display.max_colwidth", 60):
            head = df.head(50).to_string(index=False)
        return f"{len(df)} rows\n{head}"

    def _python(self, code: str) -> str:
        # A per-call print() instead of redirect_stdout: sys.stdout is process-global
        # and questions run in parallel threads.
        buf = io.StringIO()
        self.ns["print"] = functools.partial(print, file=buf)
        try:
            exec(compile(code, "<agent>", "exec"), self.ns)  # noqa: S102 - sandboxed by the container
        except Exception:  # noqa: BLE001
            buf.write(traceback.format_exc(limit=3))
        return buf.getvalue() or "(no output; use print())"

    def close(self) -> None:
        self.con.close()

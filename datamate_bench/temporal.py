"""Deterministic calendar boundaries; no question IDs or reference labels."""
import calendar
import datetime as dt
import re

from .data import fold


def month_range(year, month, last_day=None):
    end = calendar.monthrange(year, month)[1]
    return [dt.date(year, month, 1).isoformat(), dt.date(year, month, min(last_day or end, end)).isoformat()]


def context(as_of):
    day = dt.date.fromisoformat(as_of)
    monday = day - dt.timedelta(days=day.weekday())
    previous_month = day.replace(day=1) - dt.timedelta(days=1)
    return {"today": [as_of, as_of], "yesterday": [(day-dt.timedelta(days=1)).isoformat()]*2,
            "this_week": [monday.isoformat(), as_of],
            "last_week": [(monday-dt.timedelta(days=7)).isoformat(), (monday-dt.timedelta(days=1)).isoformat()],
            "this_month": month_range(day.year, day.month, day.day),
            "previous_month_same_days": month_range(previous_month.year, previous_month.month, day.day),
            "last_month": month_range(previous_month.year, previous_month.month),
            "year_to_date": [f"{day.year}-01-01", as_of]}


def resolve(question, as_of):
    q = fold(question)
    day = dt.date.fromisoformat(as_of)
    dates = context(as_of)
    full_dates=[]
    for m in re.finditer(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](20\d{2})(?!\d)",q):
        try: full_dates.append(dt.date(int(m[3]),int(m[2]),int(m[1])).isoformat())
        except ValueError: pass
    if full_dates:
        if len(full_dates)==1 and re.search(r'\btu\b.*\bden\b.*(?:hom nay|hnay|bua nay)',q):
            return [full_dates[0],as_of]
        return [full_dates[0],full_dates[1] if len(full_dates)>1 else full_dates[0]]
    # A named quarter is not a numeric month, even when written Q/Y.
    quarter=re.search(r'\bquy\s*([1-4])(?:\s*(?:/|nam\s*)\s*(20\d{2}))?',q)
    if quarter:
        first_month=(int(quarter[1])-1)*3+1
        year=int(quarter[2]) if quarter[2] else day.year
        if not quarter[2] and first_month>day.month:year-=1
        return [f'{year}-{first_month:02d}-01',month_range(year,first_month+2)[1]]
    # Yearless day/month followed by a relative endpoint takes precedence over 'today'.
    partial=re.search(r'\btu\s+(?:ngay\s+)?(\d{1,2})/(\d{1,2})(?![/\d])',q)
    if partial and re.search(r'\bden\b.*(?:hom nay|hnay|bua nay)',q):
        return [dt.date(day.year,int(partial[2]),int(partial[1])).isoformat(),as_of]
    # Explicit month/year, including common shorthand. Ambiguous input remains unresolved.
    m = re.search(r"(?:thang|th|t)\s*(\d{1,2})\s*(?:/|-|nam\s*)\s*(20\d{2}|\d{2})\b", q)
    if not m:
        m = re.search(r"(?<![\d/])(\d{1,2})[/-](20\d{2}|\d{2})(?![\d/])", q)
    if m and 1 <= int(m[1]) <= 12:
        year = int(m[2]) + (2000 if len(m[2]) == 2 else 0)
        return month_range(year, int(m[1]))
    for pattern, key in [(r"hom qua|hqua|bua qua", "yesterday"),
                         (r"tuan (?:truoc|roi)", "last_week"),
                         (r"tuan (?:nay|ni)", "this_week"),
                         (r"thang (?:nay|ni)", "this_month"),
                         (r"thang (?:truoc|lien truoc)", "last_month"),
                         (r"tu dau nam|dau nam.*toi", "year_to_date"),
                         (r"hom nay|hnay|bua nay", "today")]:
        if re.search(pattern, q):
            return dates[key]
    m = re.search(r"(\d+)\s*ngay gan nhat", q)
    if m and int(m[1]) > 0:
        return [(day-dt.timedelta(days=int(m[1])-1)).isoformat(), as_of]
    return None


def event_comparison(question):
    """An explicit N-day post-event window and its immediately preceding window."""
    q=fold(question)
    m=re.search(r'(\d+)\s*ngay ke tu\s+(?:ngay\s+)?(\d{1,2})/(\d{1,2})/(20\d{2})',q)
    if not m:
        # Anaphoric date: "Ngày D tăng giá ... N ngày kể từ ngày đó".
        date=re.search(r'\bngay\s+(\d{1,2})/(\d{1,2})/(20\d{2})',q)
        count=re.search(r'(\d+)\s*ngay ke tu\s+(?:ngay\s+)?do',q)
        if date and count and re.search(r'ngay (?:ngay )?truoc',q):
            n=int(count[1]);day=dt.date(int(date[3]),int(date[2]),int(date[1]))
            before=re.search(r'(\d+)\s*ngay (?:ngay )?truoc',q)
            previous=int(before[1]) if before else n
            if n<1 or previous<1:return None
            return ([day.isoformat(),(day+dt.timedelta(days=n-1)).isoformat()],
                    [(day-dt.timedelta(days=previous)).isoformat(),(day-dt.timedelta(days=1)).isoformat()])
    if not m or not re.search(r'ngay (?:ngay )?truoc',q): return None
    count=int(m[1]); day=dt.date(int(m[4]),int(m[3]),int(m[2]))
    if count<1: return None
    before=re.search(r'(\d+)\s*ngay (?:ngay )?truoc',q)
    previous=int(before[1]) if before else count
    return ([day.isoformat(),(day+dt.timedelta(days=count-1)).isoformat()],
            [(day-dt.timedelta(days=previous)).isoformat(),(day-dt.timedelta(days=1)).isoformat()])


def month_series(question):
    q=fold(question)
    months=[f'{int(y):04d}-{int(m):02d}' for m,y in re.findall(r'(?<![\d/])(\d{1,2})/(20\d{2})(?![\d/])',q) if 1<=int(m)<=12]
    if len(months)==2 and re.search(r'\btu\b.*\bden\b',q):
        first=dt.date.fromisoformat(months[0]+'-01');last=dt.date.fromisoformat(months[1]+'-01');result=[]
        while first<=last and len(result)<=36:
            result.append(first.strftime('%Y-%m'));first=(first.replace(day=28)+dt.timedelta(days=4)).replace(day=1)
        return result if first>last else None
    return months if len(months)>=2 else None

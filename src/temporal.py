import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

class TemporalEngine:
    """
    Engine to parse and resolve relative date/time expressions based on as_of_date
    and the solar-lunar calendar.csv according to the competition rulebook.
    """
    def __init__(self, calendar_csv_path: Optional[str] = None):
        self.calendar_df = None
        if calendar_csv_path and os.path.exists(calendar_csv_path):
            self.calendar_df = pd.read_csv(calendar_csv_path)
            self.calendar_df['date'] = pd.to_datetime(self.calendar_df['date']).dt.strftime('%Y-%m-%d')

    def resolve_relative_dates(self, as_of_date_str: str) -> Dict[str, Any]:
        """
        Resolves standard rulebook time intervals relative to as_of_date:
        - hom_nay: as_of_date
        - hom_qua: as_of_date - 1 day
        - tuan_nay: Monday of the current week up to as_of_date
        - tuan_roi: Monday to Sunday of the previous week
        - thang_nay: First day of the current month up to as_of_date (Rulebook Section 1)
        - thang_truoc: First to last day of previous month
        - tet: Solar date range of Tet holiday for this year (23 Chạp to mùng 10)
        """
        dt = datetime.strptime(as_of_date_str, "%Y-%m-%d")
        
        # Today & Yesterday
        today = dt.strftime("%Y-%m-%d")
        yesterday = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Week calculations (Monday = 0, Sunday = 6 in Python)
        weekday = dt.weekday()
        current_monday = dt - timedelta(days=weekday)
        last_week_monday = current_monday - timedelta(days=7)
        last_week_sunday = current_monday - timedelta(days=1)
        
        # Month calculations
        first_day_this_month = dt.replace(day=1)
        # Last day of previous month
        last_day_prev_month = first_day_this_month - timedelta(days=1)
        first_day_prev_month = last_day_prev_month.replace(day=1)
        
        # Year calculations
        first_day_this_year = dt.replace(month=1, day=1)
        
        # Tet resolution if calendar is available
        tet_range = self.get_tet_window(dt.year)
        
        return {
            "as_of_date": today,
            "hom_nay": today,
            "hom_qua": yesterday,
            "tuan_nay": (current_monday.strftime("%Y-%m-%d"), today),
            "tuan_roi": (last_week_monday.strftime("%Y-%m-%d"), last_week_sunday.strftime("%Y-%m-%d")),
            "thang_nay": (first_day_this_month.strftime("%Y-%m-%d"), today),
            "thang_truoc": (first_day_prev_month.strftime("%Y-%m-%d"), last_day_prev_month.strftime("%Y-%m-%d")),
            "nam_nay": (first_day_this_year.strftime("%Y-%m-%d"), today),
            "tet_season": tet_range
        }

    def get_tet_window(self, year: int) -> Optional[Tuple[str, str]]:
        """
        Calculates the Tet window for a given solar year:
        From 23 tháng Chạp (lunar year - 1) to mùng 10 tháng Giêng (lunar year).
        """
        if self.calendar_df is None:
            return None
        
        # 23 tháng Chạp of (year - 1)
        chap_23 = self.calendar_df[
            (self.calendar_df['lunar_year'] == year - 1) & 
            (self.calendar_df['lunar_month'] == 12) & 
            (self.calendar_df['lunar_day'] == 23)
        ]
        
        # Mùng 10 tháng Giêng of year
        gieng_10 = self.calendar_df[
            (self.calendar_df['lunar_year'] == year) & 
            (self.calendar_df['lunar_month'] == 1) & 
            (self.calendar_df['lunar_day'] == 10)
        ]
        
        if not chap_23.empty and not gieng_10.empty:
            start_date = chap_23.iloc[0]['date']
            end_date = gieng_10.iloc[0]['date']
            return (start_date, end_date)
        
        return None

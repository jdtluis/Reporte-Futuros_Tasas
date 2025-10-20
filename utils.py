import calendar
from datetime import datetime, timedelta

def last_working_day_of_month(month_year: str) -> str:
    # Parse the input string into month and year
    month = int(month_year[:2])
    year = int(month_year[2:])
    
    # Get the last day of the given month
    last_day = calendar.monthrange(year, month)[1]
    
    # Create a date object for the last day
    last_date = datetime(year, month, last_day)
    
    # Check if it's a weekend (Saturday or Sunday)
    if last_date.weekday() == 5:  # Saturday
        last_date -= timedelta(days=1)  # Move to Friday
    elif last_date.weekday() == 6:  # Sunday
        last_date -= timedelta(days=2)  # Move to Friday
    
    # Return the date in 'DDMMYYYY' format
    return last_date.date()


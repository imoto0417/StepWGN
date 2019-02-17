# -*- coding: utf-8 -*-

import datetime as dt
def get_today():
    today = dt.datetime.today()
    value = (today.year, today.month, today.day)
    return value

today = get_today()
print(today)

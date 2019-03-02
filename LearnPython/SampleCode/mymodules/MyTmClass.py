# -*- coding: utf-8 -*-
"""My Timer Module"""

from datetime import datetime


class TMClass:
    """My Timer Class"""

    def __init__(self):
        self.pt = datetime.today()

    def __del__(self):
        print("del")

    def get_today_time(self):
        """get time fuunction"""
        today = datetime.today()
        value = (today.year, today.month, today.day,
                 today.hour, today.minute, today.second, today.microsecond)
        return(value)

    def push_time(self):
        self.pt = datetime.today()
        return(self.pt)

    def diff_time(self):
        return(datetime.today() - self.pt)


def print_today():
    today = datetime.today()
    value = (today.year, today.month, today.day,
             today.hour, today.minute, today.second, today.microsecond)
    try:
        value
    except:
        print('error!')
    finally:
        print(value)
        print('complete!')

# -*- coding: utf-8 -*-

from datetime import datetime

class TMClass:
    def __init__(self, code, name):
        self.code = code
        self.name = name
        self.count = 0
    def get_today_1(self):
        today = datetime.today()
        value = (today.year, today.month, today.day,
            today.hour, today.minute, today.second, today.microsecond)
        self.count += 1
        return(self.code, self.name, self.count, value)
    def get_today_2(self):
        today = datetime.today()
        value = (today)
        self.count += 1
        return (self.code, self.name, self.count, str(value))

classes = []
classes.append(TMClass(1, 'kanayo'))
classes.append(TMClass(2, 'takuya'))

for tm in classes:
    for i in range(1,10,2):
        print(tm.get_today_1())
        #print(tm.get_today_2())

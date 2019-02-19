from datetime import datetime

class TMClass:
    def __init__(self, code, name):
        self.pt = datetime.today()
    def get_today_1(self):
        today = datetime.today()
        value = (today.year, today.month, today.day,
            today.hour, today.minute, today.second, today.microsecond)
        return(value)
    def get_today_2(self):
        today = datetime.today()
        value = (today)
        return (self.code, self.name, self.count, str(value))
    def push_time(self):
        self.pt = datetime.today()
        return(self.pt)
    def diff_time(self):
        return(datetime.today()-self.pt)

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

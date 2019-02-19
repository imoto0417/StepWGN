# -*- coding: utf-8 -*-

from mymodules import MyTmClass as myclass
from time import sleep

tmc = myclass.TMClass(1,'imoto')
#print(tmc.get_today_1())
#print(tmc.get_today_2())
#myclass.print_today()
print(tmc.push_time())
sleep(1)
print(tmc.diff_time())

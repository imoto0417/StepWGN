# -*- coding: utf-8 -*-

from mymodules import MyTmClass as myclass
from time import sleep

tmc = myclass.TMClass()

print(myclass.__doc__)
print(myclass.TMClass.__doc__)
print(myclass.TMClass.get_today_time.__doc__)
print(tmc.get_today_time())
myclass.print_today()
print(tmc.push_time())
sleep(1)
print(tmc.diff_time())

del tmc

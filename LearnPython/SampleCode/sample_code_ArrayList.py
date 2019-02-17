# -*- coding: utf-8 -*- 

www = []
www.append('python')
www.append('izm')
www.append('com')
www.append('/')
print(www)
print('--------------------------------')
www.insert(1, '-')
www.insert(3, '.')
print(www)
www.pop(www.index('/'))
print(www)
www.insert(0, 'http://www.')
print(www)

i = 0
www2 = []

for value in www:
    www2.append(value)
print(www2)

for i in range(len(www2)):
    print(www2[i])

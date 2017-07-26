#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt

Px=2
Py=3
Cx=3
Cy=2
Theta=[math.pi/4,math.pi*2/4,math.pi*3/4,math.pi*4/4,math.pi*5/4,math.pi*6/4,math.pi*7/4]
T1 = np.array([[1,0,-Cx], [0,1,-Cy],[0,0,1]])
T2 = np.array([[1,0,Cx], [0,1,Cy],[0,0,1]])
print(math.cos(math.radians(90)))
for item in Theta:
    R=np.array([[math.cos(item),-math.sin(item),0], [math.sin(item),math.cos(item),0],[0,0,1]])
    P=np.array([Px,Py,1]).transpose()
    print(np.dot((np.dot(np.dot(T2,R),T1)),P))


    plt.plot([np.dot(np.dot(T2,R),T1).dot(P)[0]], [np.dot(np.dot(T2,R),T1).dot(P)[1]], 'ro')

plt.plot(2,3,'bo')
plt.plot(3,2,'ks')

plt.annotate('P', xy=(1.95, 3.05), xytext=(1.7, 3.25),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.axis()
plt.show()

#print(np.dot([[1,0,-1], [2,1,-2]],[[1,0,-1], [0,1,-2], [0,1,-2]]))
#print(np.dot(R,T1))
#print(math.cos(Theta[3]))
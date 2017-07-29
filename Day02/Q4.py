# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:16:37 2017

@author: HEWEI
"""

"""Q4. Learn the Python list operations and follow the commands below:
 1.Initialize an empty list L.
 2.Add 12, 8, 9 to the list.
 3.Insert 9 to the head of the list;
 4.Double the list. (e.g. change L = [1; 2; 3] to L = [1; 2; 3; 1; 2; 3])
 5.Remove all 8 in the list.
 6.Reverse the list.
"""

L = []

L.append(12)
L.append(8)
L.append(9)
L.insert(0, 9)
L.extend(L.copy())

for item in L:
    if item == 8:
        L.remove(item)

L.reverse()

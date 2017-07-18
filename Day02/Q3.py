# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:09:44 2017

@author: HEWEI
"""

"""Q3. Read a string from console. Split the string on space delimiter (" ") 
and join using a hy-phen ("-"). 
(Example: input the string "this is a string" and output as "this-is-a-string")
"""


s=input("Please input:")
result='-'.join(s.split(" "))

print('output:'+result)
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on Mon Jul 17 17:27:32 2017

@author: HEWEI
"""


""""Q2. Read a string from console and output its length, swap its cases, 
convert it to lowercase and upper case, and reverse it."""

s=input("Please input:")

length=len(s)
print('Length:'+str(length))
print('Swap Cases:'+s.swapcase())
print('Lower Case:'+s.lower())
print('Upper Case:'+s.upper())
print('Reverse Form:'+s[::-1])
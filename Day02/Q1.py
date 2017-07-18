# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:08:18 2017

@author: HEWEI
"""

""""Q1 Calculate sum of numbers from 100-200 with while and for loop"""

def addUp_while(a,b):
    i=a
    result=0
    while i<=b:
        result+=i
        i+=1
    return result

def addUp_for(a,b):
    result=0
    for x in range(a,b+1):
        result+=x
    return result

"""Output Result"""
print('\n')
print('Result of while:'+str(addUp_while(100,200))+'\n')
print('Result of for:'+str(addUp_for(100,200))+'\n')
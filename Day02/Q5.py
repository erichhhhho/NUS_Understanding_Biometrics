# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:29:13 2017

@author: HEWEI
"""

"""Q5. A cryptarithmetic puzzle is shown below. Each letter represents a single
 numeric digit.The problem is to 
nd what each letter represents so that 
 the mathematical statement is true."""

"""Q5. Version1

    Suppose each letter can be considered as the same number i.e when A is 1, 
    B is possibly 1 too...
    
    
"""

def isPZCZ(n):

    if n%10 == (n//100)%10:
        return True
    else:
        return False

def isProductMUCHZ(pzcz, product):

    if product//10000!=0 and product//100000==0:
        if product%10 == pzcz%10 and (product//100)%10 == (pzcz//10)%10:
            return True
    else:
        return False


"""Q5. Version2

    Suppose each letter represent different number i.e A is 1, B is 2, C is 3...
    This problem become a little more complex
"""

def isPZCZ2(n):

    if n%10 == (n//100)%10 and n%10!=(n//10)%10 and n%10!=(n//1000)%10 and (n//10)%10!=(n//1000)%10:
        return True
    else:
        return False

def isProductMUCHZ2(pzcz, product):
    s=str(product)
    for item in s:
        if s.count(item)!=1:
            return False
    
    if product//10000!=0 and product//100000==0:
        if product%10 == pzcz%10 and (product//100)%10 == (pzcz//10)%10:
            return True
    else:
        return False
    
S = [(x,x*15) for x in range(1000,10000) if isPZCZ(x) and isProductMUCHZ(x,x*15)]

"""let isPZCZ(x) and isProductMUCHZ(x,x*15) be isPZCZ2(x) and isProductMUCHZ2(x,x*15) 
if the condition of the problem is like version2"""

print(S)
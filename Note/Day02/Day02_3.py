# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:19:10 2017

@author: q
"""


def addn(n):
    def f(x):
        return x + n

    return f


def gcd(a, b):
    if b > a:
        a, b = b, a
    while b > 0:
        r = a % b
        a, b = b, r

        return a


def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)


def fact(n):
    result = 1
    for x in range(1, n + 1):
        result = result * x
    return result


def both_ends(s):
    if len(s) <= 2:
        return
    else:
        return s[0] + s[1] + s[len(s) - 2] + s[len(s) - 2]
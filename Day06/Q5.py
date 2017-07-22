#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Sat Jul 22 19:51:01 2017

@author: HEWEI
"""
import  cv2
from PIL import Image
from PIL import ImageOps
import numpy
from numpy import *
from pylab import *
import matplotlib
from matplotlib import pyplot

import os
os.chdir("F:/Desktop/NUS SUMMER/HW3/hw3")

flower = Image.open("flower.bmp")

flower = ImageOps.grayscale(flower)

aflower = numpy.asarray(flower) # aflower is unit8
aflower = numpy.float32(aflower)

U,S,Vt = linalg.svd(aflower);


K = 20
Sk = numpy.diag(S[:K])
Uk = U[:, :K]
Vtk = Vt[:K, :]
aImk = numpy.dot(Uk, numpy.dot( Sk, Vtk))
Imk = Image.fromarray(aImk)

K = 50
Sk50 = numpy.diag(S[:K])
Uk50 = U[:, :K]
Vtk50 = Vt[:K, :]
aImk50 = numpy.dot(Uk50, numpy.dot(Sk50, Vtk50))
Imk50 = Image.fromarray(aImk50)

K = 100
Sk100 = numpy.diag(S[:K])
Uk100 = U[:, :K]
Vtk100 = Vt[:K, :]
aImk100 = numpy.dot(Uk100, numpy.dot(Sk100, Vtk100))
Imk100 = Image.fromarray(aImk100)

K = 200
Sk200 = numpy.diag(S[:K])
Uk200 = U[:, :K]
Vtk200 = Vt[:K, :]
aImk200 = numpy.dot(Uk200, numpy.dot(Sk200, Vtk200))
Imk200 = Image.fromarray(aImk200)

pyplot.subplot(231)
pyplot.plot(S,'b.')
pyplot.title('Singular Values')

pyplot.subplot(232)
pyplot.imshow(Image.open("flower.bmp"))
pyplot.title('Original Image'), pyplot.xticks([]), pyplot.yticks([])

pyplot.subplot(233)
pyplot.imshow(Imk)
pyplot.title('K=20'), pyplot.xticks([]), pyplot.yticks([])

pyplot.subplot(234)
pyplot.imshow(Imk50)
pyplot.title('K=50'), pyplot.xticks([]), pyplot.yticks([])

pyplot.subplot(235)
pyplot.imshow(Imk100)
pyplot.title('K=100'), pyplot.xticks([]), pyplot.yticks([])

pyplot.subplot(236)
pyplot.imshow(Imk200)
pyplot.title('K=200'), pyplot.xticks([]), pyplot.yticks([])

pyplot.savefig("Q5_HEWEI.jpg")
pyplot.show()


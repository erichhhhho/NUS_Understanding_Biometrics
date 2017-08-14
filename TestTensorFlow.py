#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

hello= tf.constant('Hellow!')
sess=tf.Session()
print(sess.run(hello))
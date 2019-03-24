#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:13:03 2019

@author: akhil
"""

#network details function

import tensorflow as tf

#convolution network

def convnet(input,filter_dim,padding,k=5,stride=2,name="convnet"):
    with tf.varibale_scope(name) as scope:
        weight=tf.get_variable("W",[k,k,input.get_shape().as_list()[-1],filter],initializer=tf.random_normal_initializer(stddev=0.02))
        bias=tf.get_variable("b"[filter_dim],initializer=tf.constant_initializer(0.0))
        conv=tf.nn.conv2d(input,weight,strides=[1,stride,stride,1],padding=padding)
        conv=tf.reshape(tf.nn.bias_add(conv,bias),conv.get_shape())
        return conv


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:13:03 2019

@author: akhil
"""

#network details function

import tensorflow as tf

#convolution network default keranal 5*5

def convnet(input,filter_dim,padding,k=5,stride=2,name="convnet"):
    with tf.varibale_scope(name) as scope:
        weight=tf.get_variable("Weight",[k,k,input.get_shape().as_list()[-1],filter],initializer=tf.random_normal_initializer(stddev=0.02))
        bias=tf.get_variable("bias",[filter_dim],initializer=tf.constant_initializer(0.0))
        conv=tf.nn.conv2d(input,weight,strides=[1,stride,stride,1],padding=padding)
        conv=tf.reshape(tf.nn.bias_add(conv,bias),conv.get_shape())
        return conv


#deconvolution network  default keranal of 4*4
def deconvnet(input,out_shape,name="deconvnet"):
    with tf.variable_scope(name) as scope:
        weight=tf.get_variable("weight",[4,4,out_shape[-1],input.get_shape().as_list()[-1]],initializer=tf.random_normal_initializer(stddev=0.02))
        bias=tf.get_variable("bias",[out_shape[-1]],initializer=tf.constant_initializer(0.0))
        deconv=tf.nn.conv2d_transpose(input,weight,output_shape=out_shape,strides=[1,2,2,1],padding="SAME")
        deconv=tf.reshape(tf.nn.bias_add(deconv,bias),deconv.get_shape())
        return deconv
    
#linear network
def linear(input,output_size,name="linear"):
    with tf.variable_scope(name) as scope:
        mat=tf.get_variable("mat",[input.get_shape().as_list()[1],output_size],tf.float32,tf.random_normal_initializer(0.2))
        bias=tf.get_variable("bias",[output_size],initializer=tf.constant_initializer(0.0))
        return tf.matmul(input,mat)+bias
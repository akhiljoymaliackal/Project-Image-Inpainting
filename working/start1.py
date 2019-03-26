#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:13:03 2019

@author: akhil
"""


#provided path for trainuing data   = ./dataset/training/*.jpg
#network details function



#importing libraries
import tensorflow as tf
from glob import glob
import os
import numpy as np
import cv2
#convolution network default keranal 5*5

#global contansts

img_height=64
img_width=64
channel=3
training_path=glob("./dataset/training/*.jpg")
bth_size=64
test_path=glob("./data/test/*.jpg")

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
    
def dilatedconv(input,out_shape,rate,name="dilatedconv"):
    with tf.variable_scope(name) as scope:
        weight=tf.get_variable("weight",[3,3,input.get_shape().as_list()[-1],out_shape[-1]])
        bias=tf.get_variable("bias",[out_shape[-1]],initializer=tf.constant_initializer(0.0))
        dilated_conv=tf.nn.atrous_conv2d(input,weight,rate=rate,padding="SAME")
        dilated_conv=tf.reshape(tf.nn.bias_add(dilated_conv,bias),dilated_conv.get_shape(0.0))
        
#normalization

def batchnorm(input,name="batchnorm"):
    with tf.variable_scope(name) as scope:
        input=tf.identity(input)
        channels=input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
		
        mean, variance = tf.nn.moments(input, axes=[0,1,2], keep_dims=False)
        normalized_batch = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=1e-5)
        return normalized_batch


#create patch and fix at a location  using tf
        
def createpatch(input):   #evaluvated and working fine plz dont touch
    margin=5
    inpimg_shape=input.get_shape().as_list()
    patch_size=tf.random_uniform([2],minval=15,maxval=25,dtype=tf.int32)
    patch=tf.zeros([patch_size[0],patch_size[1],inpimg_shape[-1]],dtype=tf.float32) #create patch with the above size
    max_x=inpimg_shape[0]-patch_size[0]-margin
    max_y=inpimg_shape[1]-patch_size[1]-margin
    x=tf.random_uniform([1],minval=margin,maxval=max_x,dtype=tf.int32)[0]
    y=tf.random_uniform([1],minval=margin,maxval=max_y,dtype=tf.int32)[0]
    dwn=inpimg_shape[0]-x-patch_size[0]
    rig=inpimg_shape[1]-y-patch_size[1]
    padding=[[x,dwn],[y,rig],[0,0]]
    masked_img = tf.pad(patch, padding, "CONSTANT", constant_values=1)
    cord=x,y  #cordinates to fix the path here we chhose random patch at random location
    result=tf.multiply(input,masked_img)
    return result,masked_img,cord,patch_size


#load the training data
    
def load_traindata():
    path=training_path
    tot_num=len(path)
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path))
    key, image_file = tf.WholeFileReader().read(filename_queue)
    images = tf.image.decode_jpeg(image_file, channels=channel)
    images = tf.image.resize_images(images ,[img_height,img_width])# size to convert need to be resized to 128
    images = tf.image.convert_image_dtype(images, dtype=tf.float32) / 127.5 - 1
    orig_images = images
    images,mask,cord,size=createpatch(images)
    mask=tf.reshape(mask,[img_height,img_width,channel]) #neesd to reshaped to 128
    mask=-(mask-1)
    images+=mask
    batch_size=bth_size
    orig_imgs, perturbed_imgs, mask, coord, pad_size = tf.train.shuffle_batch([orig_images, images, mask, cord, size], batch_size=batch_size,capacity=batch_size*2,min_after_dequeue=batch_size)
    return orig_imgs, perturbed_imgs, mask, coord, pad_size,tot_num
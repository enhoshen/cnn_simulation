#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 06:54:31 2017

@author: enhoshen
"""
import numpy as np
from parameter import *


def Unit(num):
	i=0
	_str=''
	_num=num
	_unit=[0,0,0,0,0]
	while( _num>1024):
		_unit[4-i]=_num % 1024
		_num = _num / 1024
		i += 1
	_unit[4-i]=_num
	for i , u in enumerate(_unit):
		if u != 0:
			_str+= str(u)+{0:'T',1:'G',2:'M',3:'K',4:''}[i] 
	return _str

class Layer:


	def __init__(self,in_shape,out_shape,com_arg=None):
		if com_arg != None:		
			self.type       = com_arg['type']
			self.stride     = com_arg['stride']
			self.tp_prec    = com_arg['tp_prec']
			self.fps_goal   = com_arg['fps_goal']
		else:
			self.type       = layer_com_arg_def['type']
			self.stride     = layer_com_arg_def['stride']
			self.tp_prec    = layer_com_arg_def['tp_prec']
			self.fps_goal   = layer_com_arg_def['fps_goal']
			
		self.in_h    = in_shape[0]
		self.in_w    = in_shape[1]
		self.in_ch   = in_shape[2]
		self.in_prec = in_shape[3]
		
		self.flt_h    = out_shape[0]
		self.flt_w    = out_shape[1]
		self.out_ch   = out_shape[2]
		self.out_prec = out_shape[3]
	def XB_count(self):
		o_pixel  =self.flt_h*self.flt_w*self.in_ch
		o_frame  =o_pixel*self.in_h*self.in_w/(self.stride*self.stride)
		o_volume =o_frame*self.out_ch
		precision=self.in_prec*self.out_prec
		return o_volume*precision
	def Size(self):
		weight_size=self.flt_h*self.flt_w*self.in_ch*self.out_ch*self.out_prec/8
		input_size =self.in_h*self.in_w*self.in_ch*self.in_prec/8
		return weight_size, input_size ,Unit(weight_size)+' Byte for weights; '+Unit(input_size)+' Byte for activations'
	def Do_row (self, batch ,row_cnt , flt_cnt=1 ,ch_cnt=8): #parallel do *_cnt , return number of repetition
		row_rep=self.in_h/self.stride/row_cnt
		ch_rep =self.in_ch/ch_cnt
		flt_rep=self.out_ch/flt_cnt
		weight_size=ch_cnt*flt_cnt*self.flt_w*self.flt_h/8  
		input_size =self.in_w*row_cnt*ch_cnt/8
		rep_cnt=row_rep*ch_rep*flt_rep*self.out_prec*self.in_prec
		
		return rep_cnt , {'unique_weight_size':weight_size,   
		                  'eyeriss_weight_size':weight_size*batch*row_cnt,
						  'unique_input_size' :input_size,
						  'eyeriss_input_size':input_size*self.flt_h
                         }
	def Do_pix (self):
		return 0
	def __repr__(self):
		return ''
	def __str__(self):
		return ''

class Model:
	def __init__(self, layers):
		self.layers=layers
		self.layer_size=len(layers)
	def append(self,layer):
		self.layers.append(layer)
	def XBC(self):
		XBC= [layer.XB_count() for layer in self.layers]
		return XBC
	def Size(self):
		w_size = [layer.Size()[0]for layer in self.layers]	
		in_size =  [layer.Size()[1]for layer in self.layers]	
		return w_size,in_size ,Unit(np.sum(w_size))+' Byte for weights; '+Unit(np.sum(in_size))+' Byte for activations'
		

class Architecture:

	def __init__(self, model ,shape , com_arg = None):
		
		self.pe_w= shape[0]
		self.pe_h= shape[1]
		self.gbuf_size=shape[2]
		self.bc_w= shape[3]
		self.model=model
		
	def peak_performance(self):
		return 
	def buffer_requirment(self,i):
		_w,_x=model.layers[i].Size()
		_conf =model.layers[i].conf
		#TODO
		return buf_size
	def Do_row(self , layer):
		return 0
	def compute_layer (self,i):
		#TODO
		return [off_chip,on_chip]
	def FPS_xbc(self):
		return
	def __repr__(self):
		return 0
	def __str__(self):
		return ''
def alex_model():
	l1=Layer([224,224,3,8],[11,11,64,1],{'type':'conv','stride':4,'tp_prec':12,'fps_goal':30})
	l2=Layer([55,55,64,1],[5,5,128,1],{'type':'conv','stride':2,'tp_prec':12,'fps_goal':30})
	return [l1,l2]
test=Layer([7,7,64,1],[3,3,128,1])
test2=Layer([7,7,128,1],[3,3,64,1])

Wrap=[test,test2,test2]
print test.Size()[2]
rep , _ = test.Do_row(4 , 7, 1 , 16)
print str(rep) + 'repetition'
print _
m1=Model(Wrap)
print m1.Size()


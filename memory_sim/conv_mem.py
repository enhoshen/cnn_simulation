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


	def __init__(self,in_shape,out_shape,com_arg=None,in_arg=None,flt_arg=None
	
	            ):
				
		self.type       = layer_com_arg_def['type']
		self.stride     = layer_com_arg_def['stride']
		self.tp_prec    = layer_com_arg_def['tp_prec']
		self.fps_goal   = layer_com_arg_def['fps_goal']
		
		self.in_h    = in_arg_def['in_height']
		self.in_w    = in_arg_def['in_width']
		self.in_ch   = in_arg_def['in_channel']
		self.in_prec = in_arg_def['in_prec']
		
		self.flt_h    = flt_arg_def['flt_height']
		self.flt_w    = flt_arg_def['flt_width']
		self.out_ch   = flt_arg_def['out_channel']
		self.out_prec = flt_arg_def['out_prec']
		if in_arg != None and flt_arg !=None:
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
		weight_size=self.flt_h*self.flt_w*self.in_ch*self.out_ch*self.in_prec/8
		input_size =self.in_h*self.in_w*self.in_ch*self.out_prec/8
		print(Unit(weight_size)+' Byte for weights; '+Unit(input_size)+' Byte for activations')
		return weight_size, input_size
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
		_xbc = 0
		for layer in self.layers:
			_xbc += layer.XB_count()
		#xnor bitcount counts
		return Unit(_xbc)
class Architecture:

	def __init__(self,model, conf=[14,10,'1D',5,4]):
		self.name='generated'
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
test=Layer([7,7,64,1],[3,3,128,1])
test2=Layer([7,7,128,1],[3,3,64,1])

Wrap=[test,test2]
test.Size()
rep , _ = test.Do_row(4 , 7, 1 , 16)
print str(rep) + 'repetition'
print _
m1=Model(Wrap)
print(Unit(test.XB_count()))
print(m1.XBC())

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 06:54:31 2017

@author: enhoshen
"""
import numpy as np
import * from parameter

def Unit(num):
	i=0
	_str=''
	_num=float(num)
	_unit=[0,0,0,0,0]
	while( _num>1024.0):
		_unit[4-i]=_num % 1024.0
		_num = _num/ 1024.0
		i += 1
	_unit[4-i]=_num
	for i , u in enumerate _unit:
		if u != 0:
			_str+= u+{0:'T',1:'G',2:'M',3:'K',4:''}[i]  
	return _str

class Layer:


	def __init__(self,com_arg=layer_com_arg_def,in_arg=in_arg_def,flt_arg=flt_arg_def):
		self.type       = com_arg['type']
		self.stride      = com_arg['stride']
		self.batch      = com_arg['batch']
		self.channels   = com_arg['channels']
		self.filters    = com_arg['filters']
		self.in_prec    = com_arg['in_prec']
		self.tp_prec    = com_arg['tp_prec']
		self.o_prec     = com_arg['o_prec']
		self.fps_goal   = com_arg['fps_goal']
		
		self.in_h  = in_arg['in_height']
		self.in_w  = in_arg['in_width']
		self.in_ch = in_arg['in_channel']
		
		self.flt_h = flt_arg['flt_height']
		self.flt_w = flt_arg['flt_widht']
		self.out_ch= flt_arg['out_channel']

	def XB_count(self):
		o_pixel=self.flt_h*self.flt_w*self.in_ch
		o_frame=o_pixel*self.in_h*self.in_w/(self.stride*self.stride)
		o_volume=o_frame*self.self.out_ch
		precision=self.in_prec*self.
		return o_volume*precision*self.batch
	def Size(self):
		_weight=self.filter_shape[0]*self.filter_shape[1]*self.filter_shape[2]*self.filter_shape[3]*self.filter_shape[4]/8
		_input= self.input_shape[0]*self.input_shape[1]*self.input_shape[2]*self.input_shape[3]*self.input_shape[4]/8
		print(Unit(_weight)+'B for weights; '+Unit(_input)+'B for activations')
		return [_weight , _input]
	def Do_row (self, row_cnt , flt_cnt=1 ,ch_cnt=8): #parallel do *_cnt , return number of repetition
		_row_rep=self.input_shape[1]/self.stride/row_cnt
		_ch_rep=self.input_shape[3]/ch_cnt
		_flt_rep=self.filter_shape[3]/flt_cnt
		return _row_rep*_ch_rep*_flt_rep
	def __repr__(self):
		return 0
	def __str__(self):
		return ''

class Model:
	def __init__(self, layers):
		self.layers=layers
		self.layer_size=len(layers)
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
	def FPS_xbc(self)
	
	def __repr__(self):
		return 0
	def __str__(self):
		return ''
test=Layer([1,7,7,64,1],[3,3,64,128,1])
test2=Layer([1,7,7,128,1],[3,3,64,64,1])

Wrap=[test,test2]
test.Size()
m1=Model(Wrap)
print(Unit(test.XB_count()))
print(m1.XBC())

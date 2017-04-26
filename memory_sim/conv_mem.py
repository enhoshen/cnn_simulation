#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 06:54:31 2017

@author: enhoshen
"""
import numpy as np


# tensorflow input tensor:[batch , in_height, in_width, in_channels ]
#            filter tensor:[height, width, in_channels, out_channels]
# define input shape:[batch , in_height, in_width, in_channels ]
#        filter shape:[height, width, in_channels, out_channels]

def Unit(num):
	_unit=0
	_num=float(num)
	while( _num>1024.0):
		_num = _num/ 1024.0
		_unit += 1
	return str(_num)+{0:'',1:'K',2:'M',3:'G',4:'T'}[_unit]
class Layer:
	conf_arg={ 'type':0      ,
			   'stride':1    ,
			   'batch':2     ,
			   'channels':3  ,
			   'filters':4   ,
			   'in_prec':5   ,
			   'tp_prec':6   ,
			   'o_prec' :7  ,
			   'fps_goal':30,
			   ''
	}
	tf_in_arg={'batch':0,
			   'in_height':1,
			   'in_width' :2,
			   'in_channel':3
	}
	tf_flt_arg={'height':0,
			    'width' :1,
				'in_channel':2,
				'out_channel':3
	}
	def __init__(self, input_shape ,filter_shape,conf=['conv',1,4,3,1,1,12,1]):
		self.input_shape=input_shape
		self.filter_shape=filter_shape
		self.conf=conf
	def XB_count(self):
		_batch = self.conf[self.conf_arg['batch']]
		_stride = self.conf[self.conf_arg['stride']]
		_flt_wid= self.filter_shape[self.tf_flt_arg['width']]
		_flt_height= self.filter_shape[self.tf_flt_arg['width']]
		_in_ch  = self.input_shape[self.tf_in_arg['in_channel']]
		o_pixel=_flt_wid*_flt_height*_in_ch
		o_frame=o_pixel*(self.input_shape[1]*self.input_shape[2]/_stride/_stride)
		o_volume=o_frame*self.filter_shape[3]
		_precision=self.conf[self.conf_arg['in_prec']]*self.conf[self.conf_arg['o_prec']]
		return o_volume*_precision*_batch
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
	conf_arg={ 'PE_width':0    ,
			   'PE_height':1   ,
			   'PE_type':2     ,
			   'gbuffer_size':3, #in KB
			   'bc_width':4 # in BYTE
	}
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

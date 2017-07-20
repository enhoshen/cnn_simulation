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
	_num=int(num)
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
def Units(nums):
	_str=''
	_str_list=[]
	for num in nums:
		_str_list.append(Unit(num))
		_str+=str(Unit(num))+' '
	return _str_list , _str
def str2unit(str):
	for i,c in enumerate(str):
		if c.isdigit() is False:
			break
	unit={'GB':8*1e9 ,'G':1e9,'MB':8*1e6 , 'M':1e6 ,'KB':8*1e3, 'K':1e3 ,'B':8 , '': 1}[str[i:]]
	return float(str[:i])*unit

class Layer:


	def __init__(self,in_shape,out_shape , pe_arg , pe_arr_arg ,com_arg=None):
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
		self.w_prec   = out_shape[3]
		self.out_prec = out_shape[4]
		
		self.bat_num   = pe_arg[0]
		self.flt_num =self.p= pe_arg[1]
		self.ch_num  =self.q= pe_arg[2]
		self.wb_num  = pe_arg[3]
		self.xb_num  = pe_arg[4]
		
		self.r = pe_arr_arg[0]
		self.t = pe_arr_arg[1]
		
	def XB_count(self):
		o_pixel  =self.flt_h*self.flt_w*self.in_ch
		o_frame  =o_pixel*self.in_h*self.in_w/(self.stride*self.stride)
		o_volume =o_frame*self.out_ch
		precision=self.in_prec*self.out_prec
		return o_volume*precision
	def Size(self):
		weight_size=self.flt_h*self.flt_w*self.in_ch*self.out_ch*self.w_prec/8
		input_size =self.in_h*self.in_w*self.in_ch*self.in_prec/8
		return weight_size, input_size ,Unit(weight_size)+' Byte for weights; '+Unit(input_size)+' Byte for activations'
	def Do_row (self, batch ,row_cnt, bc_w ): #parallel do *_cnt , return number of repetition
		ch_cnt = self.ch_num*self.r*bc_w
		flt_cnt= self.flt_num*self.t
		row_rep=self.in_h/self.stride/row_cnt
		ch_rep =self.in_ch/ch_cnt if self.in_ch>ch_cnt else 1 
		flt_rep=self.out_ch/flt_cnt
		bat_rep=batch/self.bat_num
		w_rep = 1 if self.w_prec == 1 else (self.w_prec-1)/self.wb_num
		x_rep = self.in_prec / self.xb_num
		# buffer requirement
		wz_size= ch_cnt*flt_cnt if self.w_prec > 1 else 0
		wb_size= ch_cnt*flt_cnt*self.wb_num
		weight_size=(wz_size+wb_size)*self.flt_w*self.flt_h/8
		
		input_size =self.in_w*row_cnt*ch_cnt*self.in_prec*self.bat_num/8
		rep_cnt=row_rep*ch_rep*flt_rep*bat_rep*w_rep*x_rep
		p_sum_size =self.in_w*row_cnt*self.bat_num*flt_cnt*self.tp_prec/8
		
		traffic1 = bat_rep*flt_rep*ch_rep*(2*p_sum_size)+bat_rep*ch_rep*(2*input_size)+bat_rep*ch_rep*flt_rep*(2*weight_size)
		# 
		traffic2 = flt_rep*ch_rep*bat_rep*(2*p_sum_size)+flt_rep*ch_rep*bat_rep*(2*input_size)+flt_rep*ch_rep*(2*weight_size)
		#
		traffic3 = bat_rep*flt_rep*(2*p_sum_size)+ bat_rep*flt_rep*ch_rep*(2*input_size)+bat_rep*flt_rep*ch_rep*(2*weight_size)
		return rep_cnt , {'unique_weight_size' :weight_size,
		                  'eyeriss_weight_size':weight_size*row_cnt,
						  'unique_input_size'  :input_size,
						  'eyeriss_input_size' :input_size*self.flt_h,  
						  'partial_sum_size'   :p_sum_size,
						  'pad_size':weight_size+input_size+p_sum_size,
						  'traffic':[traffic1,traffic2,traffic3]}
	def Do_pix (self):
		return 0
	def __repr__(self):
		return ''
	def __str__(self):
		return ''

class Model:
	def __init__(self, layers):
		self.layers=layers
		self.size=len(layers)
	def append(self,layer):
		self.layers.append(layer)
	def XBC(self):
		XBC= [layer.XB_count() for layer in self.layers]
		return XBC , np.sum(XBC)
	def Size(self):
		w_size = [layer.Size()[0]for layer in self.layers]
		in_size =  [layer.Size()[1]for layer in self.layers]
		return w_size,in_size ,Unit(np.sum(w_size))+' Byte for weights; '+Unit(np.sum(in_size))+' Byte for activations'

class PE:
	def __init__(self, buf_size , bc_w , com_arg=None  ):
		self.buf_size=str2unit(buf_size)  # in bytes
		self.throughput=self.Throughput(bc_w)
		self.bc_w=bc_w
		self.latency = self.Latency()
	def Throughput(self , bc_w): # in XB count/
		return bc_w
	def Latency(self):
		return 100
class Arch:

	def __init__(self, freq ,PE ,model ,shape , com_arg = None):

		self.f=str2unit(freq)
		self.PE = PE
		self.pe_w     =shape[0]
		self.pe_h     =shape[1]
		self.gbuf_size=str2unit(shape[2])
		self.model=model
		
	def peak_performance(self,utilize):
		pe_num=self.pe_w*self.pe_h
		peak_throughput=pe_num*self.PE.throughput
		peak_xbc=self.f*peak_throughput*utilize
		fps=peak_xbc/self.model.XBC()[1]
		return  Unit(peak_xbc) , fps
	def buffer_requirment(self):
		w_size,in_size,_ =self.model.Size()

		return None
	def Do_row(self ,N, i):
		# N is total batch number
		for _ in i:
			layer = self.model.layers[_]
			row_cnt = layer.in_w if layer.in_w <= self.pe_w else  self.pe_w
			table=layer.Do_row(N,row_cnt,self.PE.bc_w)[1]
			#print 'spad:'+Unit(table['pad_size'])+' '+'traffic:'
			print  Units(table['traffic'])[1]
			print Unit(table['unique_weight_size'])
			print Unit(table['partial_sum_size'])
			print Unit(table['unique_input_size'])
		return None 
	def compute_layer (self,i):
		return None
	def FPS_xbc(self):
		return None
	def __repr__(self):
		return None
	def __str__(self):
		return None
def alexnet_model():
	l1=Layer([55,55,3 , 1],[11,11,96,1,1],[1,16,1,1,1],[1,2])
	l2=Layer([27,27,96, 1],[5,5,256 ,1,1],[1,16,2,1,1],[1,1])
	l3=Layer([13,13,256,1],[3,3,384 ,1,1],[4,16,4,1,1],[1,4])
	l4=Layer([13,13,384,1],[3,3,384 ,1,1],[4,16,3,1,1],[2,2])
	l5=Layer([13,13,384,1],[3,3,256 ,1,1],[4,16,3,1,1],[2,2])
	return [l1,l2,l3,l4,l5]
def test_model():
	l=[]
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[1,1,1,1,1],[1,1]) )
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[1,1,8,1,1],[1,1]) )
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[1,8,1,1,1],[1,1]) )
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[8,1,1,1,1],[1,1]) )
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[1,2,4,1,1],[1,1]) )
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[1,4,2,1,1],[1,1]) )
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[2,4,1,1,1],[1,1]) )
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[4,2,1,1,1],[1,1]) )
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[4,1,2,1,1],[1,1]) )
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[2,1,4,1,1],[1,1]) )
	l.append( Layer([13,13,384,1],[3,3,384,1,1],[2,2,2,1,1],[1,1]) )
	


	return l
	
if __name__ == "__main__":

	alex=alexnet_model()
	#print test.Size()[2]
	#rep , _ = test.Do_row(4 , 7, 1 , 16)
	#print str(rep) + 'repetition'
	#print _
	m1=Model(alex)
	#print 'weights:'+Units(m1.Size()[0])[1]
	#print 'activations: '+Units(m1.Size()[1])[1]
 	#print m1.Size()[2]
	#print Unit(m1.XBC()[1])
	#print [Unit(_) for _ in m1.XBC()[0]]
	p1=PE('128B',16)
	a1=Arch( '100M' , p1, m1, [14,12,'10KB'] )
	a1.Do_row(4,range(m1.size))
	print a1.peak_performance(0.9)


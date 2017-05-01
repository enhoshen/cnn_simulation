layer_com_arg_def={ 'type':'conv'  ,  # common arguments
				'stride':1    ,
				'batch':4     ,
				'channels':64 ,
				'filters':32  ,
				'tp_prec':12  ,
				'fps_goal':30
				
		}
in_arg_def={    'batch':4,
				'in_height':7,
				'in_width' :7,
				'in_channel':32,
				'in_prec':2   
		}
flt_arg_def={       'flt_height':3,
					'flt_width' :3,
					'in_channel':32,
					'out_channel':32,
					'out_prec' :2   
		}
arch_com_arg_def={ 'PE_width':14    ,
				'PE_height':12   ,
				'PE_type':'1D'     ,
				'gbuffer_size':10, #in KB
				'bc_width':4 # bit counting operation in BYTE
		}
# tensorflow input tensor:[batch , in_height, in_width, in_channels ]
#            filter tensor:[height, width, in_channels, out_channels]
# define input shape:[batch , in_height, in_width, in_channels ]
#        filter shape:[height, width, in_channels, out_channels]
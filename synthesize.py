import numpy as np
seed = 0
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.optimizers import Adam

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu, smooth_sigmoid

import hls4ml
import argparse

def print_dict(d, indent=0):
  align=20
  for key, value in d.items():
    print('  ' * indent + str(key), end='')
    if isinstance(value, dict):
      print()
      print_dict(value, indent+1)
    else:
      print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))


# Build floating point Keras model     
def buildKeras(): 
  inputs = Input(shape=[98, ], name='input_features')
  x = Dense(20, activation='relu', name='fc1')(inputs)
  x = Dense(1, activation='sigmoid', name='output')(x)
  model = Model(inputs, x)

  optimizer = Adam(learning_rate=0.001)
  model.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])
  model.summary()

  # Synthesise


  config = hls4ml.utils.config_from_keras_model(model, granularity='model',default_reuse_factor=1, default_precision='ap_fixed<16,6>')
  config['Model']['Strategy'] = 'Latency'
  print("-----------------------------------")
  print("Configuration")
  print_dict(config)
  print("-----------------------------------")
  cfg = hls4ml.converters.create_config(backend='Vivado')
  cfg['HLSConfig']  = config
  cfg['KerasModel'] = model
  cfg['OutputDir']  = '/mnt/data/thaarres/dnd/keras_model/hls4ml_prj'
  cfg['XilinxPart'] = 'xczu9eg-ffvb1156-2-e'
  cfg['ClockPeriod'] = 10
  print_dict(cfg)
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  # hls_model = hls4ml.converters.convert_from_keras_model(model,
  #                                                        hls_config=config,
  #                                                        output_dir='/mnt/data/thaarres/dnd/keras_model/hls4ml_prj',
  #                                                        backend='Vivado',
  #                                                        part='xczu9eg-ffvb1156-2-e')                                                   
  hls_model.compile()
  hls_model.build(csim=False, synth=True, vsynth=True)

# Build quantized QKeras model    
def buildQKeras():  
  inputs = Input(shape=[98, ], name='qinput_features')
  x = QDense(20, kernel_quantizer=quantized_bits(4,0,alpha=1), bias_quantizer=quantized_bits(4,0,alpha=1), name='fc1')(inputs)
  x = QActivation(activation=quantized_relu(4), name='relu1')(x)
  x = QDense(1, kernel_quantizer=quantized_bits(4,0,alpha=1), bias_quantizer=quantized_bits(4,0,alpha=1), name='output')(x)
  # x = Activation(activation="sigmoid", name='output_act')(x)
  x = QActivation("quantized_relu(bits=16,integer=0,use_sigmoid=1)", name='output_act')(x)
  model = Model(inputs, x)  
  model.load_weights('4bitMLPmodel.h5')
  optimizer = Adam(learning_rate=0.001)
  model.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])
  model.summary()

  # Synthesise
  config = hls4ml.utils.config_from_keras_model(model, granularity='name',default_reuse_factor=1)
  config['Model']['Strategy'] = 'Latency'
  config['LayerName']['fc1']['Strategy'] = 'Resource'
  print("-----------------------------------")
  print("Configuration")
  print_dict(config)
  print("-----------------------------------")
  cfg = hls4ml.converters.create_config(backend='Vivado')
  cfg['HLSConfig']  = config
  cfg['KerasModel'] = model
  cfg['OutputDir']  = '/mnt/data/thaarres/dnd/qkeras_model/hls4ml_prj'
  cfg['Part'] = 'xc7z100-ffg900-2'
  cfg['ClockPeriod'] = 10
  print_dict(cfg)
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  # hls_model = hls4ml.converters.convert_from_keras_model(model,
  #                                                        hls_config=config,
  #                                                        output_dir='/mnt/data/thaarres/dnd/qkeras_model/hls4ml_prj',
  #                                                        backend='Vivado',
  #                                                        part='xc7z100-ffg900-2')                                                       
  hls_model.compile()
  # hls_model.build(reset=False, csim=False, synth=True, cosim=True, validation=True, export=True, vsynth=True)
  hls_model.build(csim=False, synth=True, vsynth=True)


# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-Q", "--quantize", help="Build quantized", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
  
  if args.quantize:
    buildQKeras()
  else:
    buildKeras()
      
    
    
  
                               
import  tensorflow as tf
from net import BilinearUpsampling
converter=tf.lite.TFLiteConverter.from_keras_model_file('dddtrue_weight_dsf.hdf5',input_shapes={'main_input':[1,320,240,3],},custom_objects={'BilinearUpsampling':BilinearUpsampling})
tflite_model = converter.convert()
open("xx.tflite", "wb").write(tflite_model)
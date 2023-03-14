# -*- coding: utf-8 -*-
# @Author   : WenHan
import tensorflow as tf
import tensorflow.keras as keras
import tf2onnx

model = keras.models.load_model("./model_files/alexnet_1d_ptbxl.h5")

spec = (tf.TensorSpec((None, 2048, 8), tf.float32, name="input"),)
output_path = "./model_files/alexnet_1d_ptbxl_k.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=12, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
print(output_names)
# 注意optset，因为板子上装的是onnxruntime 1.4，optset应该设置为12
# https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support


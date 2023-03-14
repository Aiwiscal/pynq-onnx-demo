# pynq-onnx-demo
在pynq上的onnx神经网络推理示例
#### 使用说明

1.  data: 存储测试数据，带k的为keras输入模式(N, L, C),不带k的为pytorch输入模式(N, C, L).其中C=8，L=2048
2.  model_files:模型文件。.pth和.h5为训练得到的pytorch和keras的模型文件；带k的onnx文件由h5转化而来，不带k的由pth转化而来
3.  models: Alexnet1d的pytorch代码
4. keras2onnx: keras .h5模型文件转化为onnx
5. torch2onnx: pytorch .pth模型文件转化为onnx
6. infer_onnx_from_keras: 运行由keras模型文件转化而来的onnx
7. infer_onnx_from_torch: 运行由pytorch模型文件转化而来的onnx

整体工程可放置于PYNQ环境下，

运行infer_onnx_from_torch.py 或 infer_onnx_from_keras.py观察推理速度。
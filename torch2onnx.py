# -*- coding: utf-8 -*-
# @Author   : WenHan

import torch
import onnx
from models.alexnet import AlexNet1d

#  把pth模型转化为onnx

if __name__ == '__main__':
    my_model = AlexNet1d()
    my_model.eval()
    my_model.load_state_dict(torch.load("./model_files/alexnet_1d_ptbxl.pth"))
    input_names = ['input']
    output_names = ['output']

    x = torch.randn(1, 8, 2048)
    torch.onnx.export(my_model, x, './model_files/alexnet_1d_ptbxl.onnx', input_names=input_names, output_names=output_names,
                      verbose='True')

    model = onnx.load("./model_files/alexnet_1d_ptbxl.onnx")

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))

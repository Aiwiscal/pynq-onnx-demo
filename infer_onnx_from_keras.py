# -*- coding: utf-8 -*-
# @Author   : WenHan
import time
import numpy as np
import onnxruntime as ort

if __name__ == '__main__':
    ort_session = ort.InferenceSession("./model_files/alexnet_1d_ptbxl_k.onnx")
    test_data = np.load("./data/test_data_k.npy")
    total_time = 0
    for i in range(test_data.shape[0]):
        sample = np.expand_dims(test_data[i], axis=0)
        tic = time.time()
        outputs = ort_session.run(
            None,
            {"input": sample.astype(np.float32)},
        )
        toc = time.time()
        # print(outputs[0])
        total_time += (toc - tic)

    print("average inference time (sec): ", total_time/test_data.shape[0])


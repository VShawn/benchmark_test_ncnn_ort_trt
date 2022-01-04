import os.path
from os import listdir
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms


from MobileNetV3 import get_mobilenetv3_large

# model = get_mobilenetv3_large()
model = torchvision.models.mobilenet_v3_large().cuda()



dummy_input = torch.randn(1, 3, 224, 224)
# model = torchvision.models.resnet18()
torch.onnx.export(model,               # model being run
                input,                         # model input (or a tuple for multiple inputs)
                "tewt.onnx",   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                )
import time
import onnxruntime
import onnx
from torch.autograd import Variable
import torch.onnx
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from yolox.exp import get_exp

g_gpu = True

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ])  # 转换
# open image and resize to 224, convert to tensor

# pretrained = "yolox-tiny"
pretrained = "yolox-s"


def open_image(image_path, size=640):
    image = Image.open(image_path)
    image = image.resize((size, size))
    image = transform(image)
    image = image.unsqueeze(0)
    if g_gpu:
        image = image.cuda()
    return image


# Model
model = get_exp(exp_name=pretrained).get_model()
model.cuda()
model.eval()

# Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
img = 'D:\cars.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# x = torch.rand(1, 3, 224, 224).to(device)
# x = open_image('../cat.1.jpg') # 281: '猫, tabby, tabby cat'
# x = open_image('../cat.1084.jpg') # 284 n02123597 猫, Siamese cat, Siamese
x = open_image('../dog.310.jpg')  # 250 n02110185 狗, Siberian husky
# x = open_image('../dog.823.jpg') # 263 n02113023 狗, Pembroke, Pembroke Welsh corgi
predictions = model(x)
print(predictions[0][0])


dummy_input = x

# # yolo5 官方模型转化方法:
# # https://github.com/ultralytics/yolov5/issues/251
# # ### ********************  pytorch to onnx  ******************** ###
# # dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
# torch.onnx.export(model,
#                   dummy_input,
#                   pretrained + ".onnx",
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names=['input'],   # the model's input names
#                   output_names=['output'],  # the model's output names
#                   opset_version=11,
#                   )

### ********************  test onnx  ******************** ###
print("onnx", onnxruntime.get_device())

onnx_model = onnx.load(pretrained + ".onnx")  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
session = onnxruntime.InferenceSession(pretrained + ".onnx", providers=['CUDAExecutionProvider'])
# session = onnxruntime.InferenceSession(pretrained + ".onnx", providers=['CPUExecutionProvider'])
# session = onnxruntime.InferenceSession(pretrained + ".onnx", None)
input_name = session.get_inputs()[0].name
orig_result = session.run([], {input_name: dummy_input.cpu().data.numpy()})
print("onnx out", orig_result[0][0])

### ********************  time usage  ******************** ###
time_start = time.time()

for i in range(100):
    predictions = model(x)

time_end = time.time()
t = (time_end-time_start)
print('pytorch', x.shape, ' time avg-cost =', t / 100, 's (', 100 / t, 'fps)')


data = dummy_input.cpu().data.numpy()
time_start = time.time()

for i in range(100):
    orig_result = session.run([], {input_name: data})

time_end = time.time()
t = (time_end-time_start)
print('onnx-runtime', x.shape, ' time avg-cost =', t / 100, 's (', 100 / t, 'fps)')

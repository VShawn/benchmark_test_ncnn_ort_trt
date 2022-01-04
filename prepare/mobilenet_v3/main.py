from torch.autograd import Variable
import torch.onnx
import torchvision
from PIL import Image
from MobileNetV3 import *
import torchvision.transforms as transforms
from MobileNetV3 import get_mobilenetv3_large
from model import *
import numpy as np

g_gpu = True
# g_gpu = False

# transform = transforms.Compose([transforms.ToTensor()])  # 转换

# ImageNet
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),])  # 转换
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),])  # 转换

# CIFAR100
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5071, 0.4865, 0.4409), std = (0.2673, 0.2564, 0.2762)),])  # 转换

# open image and resize to 224, convert to tensor
def open_image(image_path, size = 224):
    image = Image.open(image_path)
    image = image.resize((size, size))
    image = transform(image)
    image = image.unsqueeze(0)
    if g_gpu:
        image = image.cuda()
    return image

pretrained = 'mobilenetv3-large-1cd25616.pth'
model = get_mobilenetv3_large(from_pretrained=pretrained).cuda()
model.eval()
if g_gpu:
    model = model.cuda()




# x = torch.rand(1, 3, 224, 224).to(device)
# x = open_image('../cat.1.jpg') # 281: '猫, tabby, tabby cat'
# x = open_image('../cat.1084.jpg') # 284 n02123597 猫, Siamese cat, Siamese
x = open_image('../dog.310.jpg') # 250 n02110185 狗, Siberian husky
# x = open_image('../dog.823.jpg') # 263 n02113023 狗, Pembroke, Pembroke Welsh corgi

### ********************  test pytorch  ******************** ###
predictions = model(x)
pred = torch.max(predictions, 1)[1]
print(pred)
print("pytorch out :",predictions[0][pred.item()])

# Print top categories per image
probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
# Get imagenet class mappings
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(top5_prob.size(0)):
        print("TOP {0}({1}): ".format(i, top5_prob[i].item()), top5_catid[i], categories[top5_catid[i]])





dummy_input = x
# ### ********************  pytorch to onnx  ******************** ###
# # dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
# torch.onnx.export(model,
#                     dummy_input,
#                     pretrained + ".onnx",
#                     export_params=True,        # store the trained parameter weights inside the model file
#                     do_constant_folding=True,  # whether to execute constant folding for optimization
#                     input_names = ['input'],   # the model's input names
#                     output_names = ['output'], # the model's output names
#                 )

### ********************  test onnx  ******************** ###
import onnx
import onnxruntime
onnx_model = onnx.load(pretrained + ".onnx")  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
session = onnxruntime.InferenceSession(pretrained + ".onnx", providers=['CUDAExecutionProvider'])
# session = onnxruntime.InferenceSession(pretrained + ".onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
orig_result = session.run([], {input_name: dummy_input.cpu().data.numpy()})
print("onnx runtime out :", orig_result[0][0][pred.item()])
probabilities = torch.nn.functional.softmax(torch.from_numpy(orig_result[0][0]), dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
sum = sum(np.exp(orig_result[0][0]))
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(top5_prob.size(0)):
        print("TOP {0}({1}): ".format(i, np.exp(orig_result[0][0][top5_catid[i]]) / sum), top5_catid[i], categories[top5_catid[i]])


### ********************  test onnx sim  ******************** ###
import onnx
import onnxruntime
onnx_model = onnx.load(pretrained + ".sim.onnx")  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
session = onnxruntime.InferenceSession(pretrained + ".sim.onnx", providers=['CUDAExecutionProvider'])
# session = onnxruntime.InferenceSession(pretrained + ".sim.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
orig_result = session.run([], {input_name: dummy_input.cpu().data.numpy()})
print("sim onnx runtime out :", orig_result[0][0][pred.item()])
probabilities = torch.nn.functional.softmax(torch.from_numpy(orig_result[0][0]), dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
sum = np.sum(np.exp(orig_result[0][0]))
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(top5_prob.size(0)):
        print("TOP {0}({1}): ".format(i, np.exp(orig_result[0][0][top5_catid[i]]) / sum), top5_catid[i], categories[top5_catid[i]])



### ********************  time usage  ******************** ###
import time
time_start=time.time()

for i in range(100):
    predictions = model(x)

time_end=time.time()
t  = (time_end-time_start)
print('pytorch', x.shape, ' time avg-cost =', t / 100,'s (', 100 / t, 'fps)')



data = dummy_input.cpu().data.numpy()
time_start=time.time()

for i in range(100):
    orig_result = session.run([], {input_name: data})

time_end=time.time()
t  = (time_end-time_start)
print('onnx-runtime', x.shape, ' time avg-cost =', t / 100,'s (', 100 / t, 'fps)')
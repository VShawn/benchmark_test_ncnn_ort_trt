from torch.autograd import Variable
import torch.onnx
import torchvision
from PIL import Image
from MobileNetV3 import *
import torchvision.transforms as transforms
from MobileNetV3 import get_mobilenetv3_large
from model import *

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


# model = torchvision.models.alexnet().cuda()
model = get_mobilenetv3_large(from_pretrained='mobilenetv3-large-1cd25616.pth').cuda()

# error Exporting the operator hardsigmoid to ONNX opset version 11 is not supported.
# model_pretrained = 'mobilenet_v3_large-8738ca79.pth'
# model = torchvision.models.mobilenet_v3_large()
# model.load_state_dict(torch.load(model_pretrained))

# model = MobileNetV3(num_classes = 100)
# model.load_state_dict(torch.load('best_model_LARGE_ckpt.t7')['model'])

model.eval()
if g_gpu:
    model = model.cuda()

# x = torch.rand(1, 3, 224, 224).to(device)
# x = open_image('../cat.1.jpg') # 281: '猫, tabby, tabby cat'
x = open_image('../cat.1084.jpg') # 284 n02123597 猫, Siamese cat, Siamese
# x = open_image('../dog.310.jpg') # 250 n02110185 狗, Siberian husky
# x = open_image('../dog.823.jpg') # 263 n02113023 狗, Pembroke, Pembroke Welsh corgi


predictions = model(x)
pred = torch.max(predictions, 1)[1]
print(pred)
print(predictions[0][pred.item()])

# Print top categories per image
probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
# Get imagenet class mappings
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(top5_prob.size(0)):
        print(top5_catid[i], categories[top5_catid[i]], top5_prob[i].item())



# dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
# torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True)
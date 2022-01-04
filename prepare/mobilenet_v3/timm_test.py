import timm

model = timm.create_model('mobilenetv3_large_100', pretrained=True)
model.eval()


import urllib
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

img = Image.open('../dog.310.jpg').convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)

# Get imagenet class mappings
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
# prints class names and probabilities like:
# [('Samoyed', 0.6425196528434753), ('Pomeranian', 0.04062102362513542), ('keeshond', 0.03186424449086189), ('white wolf', 0.01739676296710968), ('Eskimo dog', 0.011717947199940681)]


from torch.autograd import Variable
import torch.onnx
dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
torch.onnx.export(model.cuda(), dummy_input, "m1.onnx", verbose=True)



RuntimeError: Exporting the operator hardsigmoid to ONNX opset version 9 is not supported. Please feel free to request support or submit a pull 
request on PyTorch GitHub.
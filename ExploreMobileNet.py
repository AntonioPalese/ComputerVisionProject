import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)

print(list(model.features.children())[-1])


# Classe mia in cui ho tutta la MobileNet tranne il classificatore finale (MLP)
class MobileNetConvLayer(nn.Module):
    def __init__(self):
        super(MobileNetConvLayer, self).__init__()
        self.features = nn.Sequential(
            # stop at last conv
            *list(model.features.children())
        )

    def forward(self, x):
        x = self.features(x)
        return x

im = Image.open("doberman.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = preprocess(im)
image_batch = image.unsqueeze(0)

model_conv = MobileNetConvLayer()
out = model_conv(image_batch)
print(out.shape)

# dimensione di out flattened : 1280 * 7 * 7 = 62720

print(out.view(-1))

#62720


import matplotlib.pyplot as plt
import torch
model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
model.eval()
import wget
import numpy as np
import cv2


from PIL import Image
from torchvision import transforms
filename = "C:\\Users\\pales\\Desktop\\CV\\firstcnn\\dataset_test\\denim\\42654509kw_1_f.jpg"
import pathlib

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_batch = torch.rand(1, 3, 224, 224)
root=pathlib.Path("dataset_test/denim")
trans = transforms.ToTensor()
for i,  files in enumerate(root.iterdir()):
    newimg = preprocess(Image.open(files))
    newimg = newimg.unsqueeze(0)

    input_batch = torch.cat((input_batch, newimg), 0)

input_batch = input_batch[1:]
 # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output, dim=0)
print(probabilities)

wget.download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(f"Immagine {i}  : ")
    for j in range(top5_prob.size(1)):
        print(categories[top5_catid[i,j]], top5_prob[i,j].item())
    print('\n')
    cv2.imshow(f"{i}", input_batch[i].permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
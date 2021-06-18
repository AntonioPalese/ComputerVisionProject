import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import pathlib
from PIL import Image
from torch.optim  import Adam
from torch.autograd import Variable

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

transformer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #NUMPY TO TENSOR
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #0-1 to [-1,1] (x-media)/std
])
train_path='C:\\Users\\pales\\Desktop\\CV\\firstcnn\\dataset_train'
test_path='C:\\Users\\pales\\Desktop\\CV\\firstcnn\\dataset_test'


train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=21, shuffle=True
)

test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=21, shuffle=True
)

#categorie

root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)
['denim','pants','shirts','tops']


# CNN Network

def conv3x3(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=3, padding=1)


def max_pool_2d():
    return nn.MaxPool2d(kernel_size=2, stride=2)


class VGGlayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, activated=True,
                 max_pool=False):
        super(VGGlayer, self).__init__()

        layers = [
            conv3x3(in_channels, out_channels),
            nn.ReLU(True),
        ]

        if max_pool:
            layers += [max_pool_2d()]

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class ConvNet(nn.Module):

    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super(ConvNet, self).__init__()

        self.conv_features = nn.Sequential(
            VGGlayer(in_channels, 64),
            VGGlayer(64, 64, max_pool=True),
            VGGlayer(64, 128),
            VGGlayer(128, 128, max_pool=True),
            VGGlayer(128, 256),
            VGGlayer(256, 256),
            VGGlayer(256, 256, max_pool=True),
            VGGlayer(256, 512),
            VGGlayer(512, 512),
            VGGlayer(512, 512, max_pool=True),
            VGGlayer(512, 512),
            VGGlayer(512, 512),
            VGGlayer(512, 512, max_pool=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_features(x)
        x = torch.flatten(x, 1)
        y = x.detach().cpu()
        x = self.classifier(x)
        return x, y

    #def __init__(self, num_classes=4):
        # super(ConvNet, self).__init__()
        #
        # # Output size after convolution filter
        # # ((w-f+2P)/s) +1
        #
        # # Input shape= (24,3,150,150)
        #
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        #
        # # Shape= (24,12,150,150)
        # self.bn1 = nn.BatchNorm2d(num_features=12)
        # # Shape= (24,12,150,150)
        # self.relu1 = nn.ReLU()
        # # Shape= (24,12,150,150)
        #
        # self.pool = nn.MaxPool2d(kernel_size=2)
        # # Reduce the image size be factor 2
        # # Shape= (24,12,75,75)
        #
        # self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # # Shape= (24,20,75,75)
        # self.relu2 = nn.ReLU()
        # # Shape= (24,20,75,75)
        #
        # self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # # Shape= (24,32,75,75)
        # self.bn3 = nn.BatchNorm2d(num_features=32)
        # # Shape= (24,32,75,75)
        # self.relu3 = nn.ReLU()
        # # Shape= (24,32,75,75)
        #
        # self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)

    # def forward(self, input):
    #     output = self.conv1(input)
    #     output = self.bn1(output)
    #     output = self.relu1(output)
    #
    #     output = self.pool(output)
    #
    #     output = self.conv2(output)
    #     output = self.relu2(output)
    #
    #     output = self.conv3(output)
    #     output = self.bn3(output)
    #     output = self.relu3(output)
    #
    #     # Above output will be in matrix form, with shape (256,32,75,75)
    #
    #     output = output.view(-1, 32 * 75 * 75)
    #
    #     output = self.fc(output)
    #
    #     return output

model=ConvNet(num_classes=4).to(device)

num_params = sum([np.prod(p.shape) for p in model.parameters()])

optimizer = Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()

num_epochs=10

train_count=len(glob.glob(train_path+'/**/*.jpg'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))

print(train_count,test_count)

best_accuracy=0.0
feature_vecs = []
j = 0
for epoch in range(num_epochs):

    #Evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0

    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())

        optimizer.zero_grad()

        outputs, _ =model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()


        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)

        train_accuracy+=int(torch.sum(prediction==labels.data))

    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count


    # Evaluation on testing dataset
    model.eval()


    test_accuracy=0.0

    for i, (images,labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
        j += 1
        outputs, feature_vec =model(images)

        for feat in range(feature_vec.shape[0]):
            feature_vecs.append(feature_vec[feat])

        _,prediction=torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))

    test_accuracy=test_accuracy/test_count


    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))

    #Save the best model
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict(),'best_checkpoint.model')
        best_accuracy=test_accuracy

effective_vec = feature_vecs[-102:];

for f in effective_vec[0]:
    print(f)
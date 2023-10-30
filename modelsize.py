import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch


class MnistNet(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = MnistNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
summary(model.to(device), input_size=(1, 28, 28))
import Net.Resnet
model = Net.Resnet.ResNet18()
summary(model.to(device), input_size=(3, 32, 32))
import Net.Resnet_new
model = Net.Resnet_new.ResNet34()
summary(model.to(device), input_size=(3, 64, 64))
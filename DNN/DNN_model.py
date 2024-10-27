import torch
import torch.nn as nn

class DNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DNNModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 512)
        self.layer4 = nn.Linear(512, 256)
        self.layer5 = nn.Linear(256, 128)
        self.layer6 = nn.Linear(128, 64)
        self.layer7 = nn.Linear(64, 32)
        self.layer8 = nn.Linear(32, 16)
        self.layerlast = nn.Linear(16, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.relu(self.layer4(x))
        x = self.dropout(x)
        x = self.relu(self.layer5(x))
        x = self.dropout(x)
        x = self.relu(self.layer6(x))
        x = self.dropout(x)
        x = self.relu(self.layer7(x))
        x = self.dropout(x)
        x = self.relu(self.layer8(x))
        x = self.layerlast(x)
        return x

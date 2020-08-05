import torch
import torch.nn as nn
import torch.nn.functional as F
"""
24C5 means a convolution layer with 24 feature maps using a 5x5 filter and stride 1
24C5S2 means a convolution layer with 24 feature maps using a 5x5 filter and stride 2
P2 means max pooling using 2x2 filter and stride 2
256 means fully connected dense layer with 256 units
"""

cfg = {
    'net_1': [32, 'MaxPool'],
    'net_1_double': [32, 32, 'MaxPool'],
    'net_1_triple': [32, 32, 32, 'MaxPool'],
    'net_2': [32, 'MaxPool', 64, 'MaxPool'],
    'net_2_double': [32, 32, 'MaxPool', 64, 64, 'MaxPool'],
    'net_2_triple': [32, 32, 32, 'MaxPool', 64, 64, 64, 'MaxPool'],
    'net_3': [32, 'MaxPool', 64, 'MaxPool', 96, 'MaxPool'],
    'net_3_double': [32, 32, 'MaxPool', 64, 64, 'MaxPool', 128, 128, 'MaxPool'],
    'net_3_triple': [32, 32, 32, 'MaxPool', 64, 64, 64, 'MaxPool', 128, 128, 128, 'MaxPool']
}

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


class PlaNet(nn.Module):
    def __init__(self, net_name, batch_norm = True, input_size = (1,2,128,128)):
        super(PlaNet, self).__init__()
        self.features = self._make_feature_extractor(net_name, batch_norm, input_size)
        flatten_features_size = self.num_flat_features(self.features(torch.rand(input_size)))
        self.predict = nn.Linear(flatten_features_size, 256)
        self.predict2 = nn.Linear(256, 1)
        print(' ')

    def forward(self, x):
        # extract features
        x = self.features(x)
        # dropout
        #x = self.dropout(x)
        # flatten
        x = x.view(-1, self.num_flat_features(x))
        # classify
        x = self.predict(x)
        x = self.predict2(x)
        return x

    def _make_feature_extractor(self, net_name, batch_norm, input_size):
        layers = []
        in_channels = input_size[1]
        layer_key_list = cfg['net_2_double']
        for l in layer_key_list:
            if l == 'MaxPool':
                layers += [nn.MaxPool2d(2, 2)]
            else:
                layers += [nn.Conv2d(in_channels, l, 3, bias=False)]
                if batch_norm:
                    layers += [nn.BatchNorm2d(l), nn.ReLU(inplace=True)]
                else:
                    layers += [nn.ReLU(inplace=True)]
                in_channels = l
        return nn.Sequential(*layers)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MLP_1(nn.Module):
    """
    (1x128x128) => 1024-RLU => 256-RLU => 29
    # [20,   250] loss: 0.528
    """
    def __init__(self,input_size):
        super(MLP_1, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.fc1_1 = nn.Linear(input_size, 1)
        self.fc1_3 = nn.Linear(5777, 1024)
        self.fc2 = nn.Linear(4000, 128)
        self.fc3 = nn.Linear(1000, 128)
        self.fc4 = nn.Linear(128,1)

    def forward(self, x):
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # (1x128x128) = > 256 - RLU
        x = self.fc1_1(x)
        # x = F.relu(x)
        # x = self.fc1_3(x)
        # x = F.relu(x)
        # #  (256) = > 128 - RLU
        # x = self.fc2(x)
        # x = F.relu(x)
        # # (128) => 29 - SOFTMAX
        #x = self.fc3(x)
        # x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Arch2(nn.Module):
    """
    (1x128x128) => 1024-RLU => 256-RLU => 29
    # [20,   250] loss: 0.528
    """
    def __init__(self,input_size):
        super(Arch2, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.fc1 = nn.Linear(input_size, 4000)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc3 = nn.Linear(1000, 128)
        self.fc4 = nn.Linear(128,1)

    def forward(self, x):
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # (1x128x128) = > 256 - RLU
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Arch3(nn.Module):
    """
    (1x128x128) => 1024-RLU => 256-RLU => 29
    # [20,   250] loss: 0.528
    """
    def __init__(self,input_size):
        super(Arch3, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.fc1 = nn.Linear(input_size, 2000)
        self.fc2 = nn.Linear(2000, 256)
        self.fc3 = nn.Linear(256,1)

    def forward(self, x):
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # (1x128x128) = > 256 - RLU
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.abs(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Arch4(nn.Module):
    """
    (1x128x128) => 1024-RLU => 256-RLU => 29
    # [20,   250] loss: 0.528
    """
    def __init__(self,input_size):
        super(Arch4, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.fc1 = nn.Linear(input_size, 4000)
        self.fc2 = nn.Linear(4000, 1)

    def forward(self, x):
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # (1x128x128) = > 256 - RLU
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.abs(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Arch5(nn.Module):
    """
    (1x128x128) => 1024-RLU => 256-RLU => 29
    # [20,   250] loss: 0.528
    """
    def __init__(self,input_size):
        super(Arch5, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 29)
        self.fc3 = nn.Linear(29,1)

    def forward(self, x):
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # (1x128x128) = > 256 - RLU
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.abs(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Arch6(nn.Module):
    """
    (1x128x128) => 1024-RLU => 256-RLU => 29
    # [20,   250] loss: 0.528
    """
    def __init__(self,input_size):
        super(Arch5, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.fc1 = nn.Linear(input_size, 3000)
        self.fc2 = nn.Linear(128, 29)
        self.fc3 = nn.Linear(29,1)

    def forward(self, x):
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # (1x128x128) = > 256 - RLU
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.abs(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

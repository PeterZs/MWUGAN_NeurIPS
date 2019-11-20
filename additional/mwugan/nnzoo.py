import torch
from torch import nn
from torch.nn import functional as F
import logging
import tqdm
import os


class Generator_gan(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator_gan, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.z_dim = input_size

    def forward(self, x):
        # x = F.leaky_relu(self.map1(x), 0.5)
        # x = F.leaky_relu(self.map2(x), 0.5)

        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        return self.map3(x)


class Discriminator_gan(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator_gan, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = F.leaky_relu(self.map1(x), 0.5)
        # x = F.leaky_relu(self.map2(x), 0.5)

        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        return torch.sigmoid(self.map3(x))




def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# G(z)
class Generator_dcgan(nn.Module):
    # initializers
    def __init__(self, dim_in, hidden_size=128, channel=1):
        super(Generator_dcgan, self).__init__()
        d = hidden_size
        self.dim_in = dim_in
        self.deconv1 = nn.ConvTranspose2d(dim_in, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        # self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d*2, channel, 4, 2, 1)
        self.weight_init(0, 0.02)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        # print('deconv1', x.shape)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        # print('deconv2', x.shape)
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        # print('deconv3', x.shape)
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        # print('deconv4', x.shape)


        return x

# G(z)
class Generator_dcgan_adastyle(nn.Module):
    # initializers
    def __init__(self, dim_in, hidden_size=128, channel=1):
        super(Generator_dcgan_adastyle, self).__init__()

        # Input: [batch_size, dim_in, 1, 1]
        # Output: [batch_size, 1, 32, 32]
        d = hidden_size
        self.d = d
        self.dim_in = dim_in
        self.linear = nn.Linear(dim_in, d * 16 * 8 * 8)
        self.deconv0_bn = nn.BatchNorm2d(d * 16)                    # [batch_size, 1, 8, 8]
        self.deconv1 = nn.ConvTranspose2d(d * 16, d*8, 4, stride=2, padding=1)
        self.deconv1_bn = nn.BatchNorm2d(d*8)                       # [batch_size, d * 8, 16, 16]
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, stride=2, padding=1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)                       # [batch_size, d * 4, 32, 32]
        # self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        # self.deconv3_bn = nn.BatchNorm2d(d*2)
        # self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d*4, channel, 1, stride=1, padding=0)
        # self.deconv5_bn = nn.BatchNorm2d(channel)                   # [batch_size, channel, 32, 32]
        self.weight_init(0, 0.02)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        # print(input.shape)
        x = self.linear(input.view(-1, self.dim_in)).view(-1, self.d * 16, 8, 8)
        x = F.leaky_relu(self.deconv0_bn(x))
        # print(x.shape)
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(x)))
        # print(x.shape)
        # print('deconv1', x.shape)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)))
        # print(x.shape)
        # print('deconv2', x.shape)
        # x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)))
        # print('deconv3', x.shape)
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        # print('deconv4', x.shape)

        return x

class Discriminator_dcgan(nn.Module):
    # initializers
    def __init__(self, hidden_size=128, channel=1, wgan=False):
        super(Discriminator_dcgan, self).__init__()
        d = hidden_size
        self.conv1 = nn.Conv2d(channel, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        # self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        # self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*4, 1, 4, 1, 0)
        self.weight_init(0, 0.02)
        self.wgan = wgan

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        # print('conv1', x.shape)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        # print('conv2', x.shape)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # print('conv3', x.shape)
        # x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        # print('conv4', x.shape)
        if self.wgan:
            return x

        x = torch.sigmoid(self.conv5(x))
        # print('conv5', x.shape)

        return x

class Discriminator_dcgan_adastyle(nn.Module):
    # initializers
    def __init__(self, hidden_size=128, channel=1, wgan=False):
        super(Discriminator_dcgan_adastyle, self).__init__()
        d = hidden_size
        self.d = d
        self.conv1 = nn.Conv2d(channel, d, 2, stride=2, padding=0)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d*2, 2, stride=2, padding=0)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 2, stride=2, padding=0)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        # self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        # self.conv4_bn = nn.BatchNorm2d(d*8)
        # self.conv5 = nn.Conv2d(d*4, 1, 4, 1, 0)
        self.linear = nn.Linear(d * 4 * (4 * 4), 1)
        self.weight_init(0, 0.02)
        self.wgan = wgan

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_bn(self.conv1(input)), 0.2)
        # print('conv1', x.shape)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        # print('conv2', x.shape)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # print('conv3', x.shape)
        # x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        # print('conv4', x.shape)
        if self.wgan:
            return x
        #print(x.shape)
        x = torch.sigmoid(self.linear(x.view(-1, self.d * 4 * (4 * 4))))
        # print('conv5', x.shape)

        return x


# G(z)
class Generator_dcgan64(nn.Module):
    # initializers
    def __init__(self, dim_in, hidden_size=128, channel=1):
        super(Generator_dcgan64, self).__init__()
        d = hidden_size
        self.dim_in = dim_in
        self.deconv1 = nn.ConvTranspose2d(dim_in, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, channel, 4, 2, 1)
        self.weight_init(0, 0.02)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x

class Discriminator_dcgan64(nn.Module):
    # initializers
    def __init__(self, hidden_size=128, channel=1, wgan=False):
        super(Discriminator_dcgan64, self).__init__()
        d = hidden_size
        self.conv1 = nn.Conv2d(channel, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        self.weight_init(0, 0.02)
        self.wgan = wgan

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        if self.wgan:
            return x
        x = torch.sigmoid(self.conv5(x))
        return x




class MNISTClassifier(nn.Module):
    def __init__(self, input_channel=1, hidden_channel=128):
        super(MNISTClassifier, self).__init__()
        # self.conv1 = nn.Conv2d(input_channel, hidden_channel, 4, 2, 1)
        self.conv2 = nn.Conv2d(input_channel, hidden_channel * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(hidden_channel * 2)
        self.conv3 = nn.Conv2d(hidden_channel * 2, hidden_channel * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(hidden_channel * 4)
        self.conv4 = nn.Conv2d(hidden_channel * 4, hidden_channel * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(hidden_channel * 8)
        self.conv5 = nn.Conv2d(hidden_channel * 8, 1, 4, 1, 0)

    def forward(self, input):
        # x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.relu(self.conv2_bn(self.conv2(input)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.sigmoid(self.conv5(x))
        return x

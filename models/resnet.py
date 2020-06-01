'''
Modified from https://github.com/jack-willturner/DeepCompression-PyTorch

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Shortcut(nn.Module):
    def __init__(self, in_planes, planes, expansion=1, kernel_size=1, stride=1, bias=False):
        super(Shortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, expansion*planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.mask1 = nn.Conv2d(in_planes, expansion*planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())
        self.bn1 = nn.BatchNorm2d(expansion*planes)

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight,  self.mask1.weight)
        return self.bn1(self.conv1(x))

    def __prune__(self, threshold):
        self.mask1.weight.data = torch.gt(torch.abs(self.conv1.weight), threshold).float()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mask1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask2.weight.data = torch.ones(self.mask2.weight.size())
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Shortcut(in_planes, planes, self.expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight,  self.mask1.weight)
        self.conv2.weight.data = torch.mul(self.conv2.weight,  self.mask2.weight)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def __prune__(self, threshold):
        self.mask1.weight.data = torch.gt(torch.abs(self.conv1.weight), threshold).float()
        self.mask2.weight.data = torch.gt(torch.abs(self.conv2.weight), threshold).float()

        if isinstance(self.shortcut, Shortcut):
            self.shortcut.__prune__(threshold)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.mask1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mask2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mask2.weight.data = torch.ones(self.mask2.weight.size())
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.mask3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.mask3.weight.data = torch.ones(self.mask3.weight.size())
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Shortcut(in_planes, planes, self.expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight,  self.mask1.weight)
        self.conv2.weight.data = torch.mul(self.conv2.weight,  self.mask2.weight)
        self.conv3.weight.data = torch.mul(self.conv3.weight,  self.mask3.weight)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def __prune__(self, threshold):
        self.mask1.weight.data = torch.gt(torch.abs(self.conv1.weight), threshold).float()
        self.mask2.weight.data = torch.gt(torch.abs(self.conv2.weight), threshold).float()
        self.mask3.weight.data = torch.gt(torch.abs(self.conv3.weight), threshold).float()

        if isinstance(self.shortcut, Shortcut):
            self.shortcut.__prune__(threshold)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, data='cifar10'):
        super(ResNet, self).__init__()
        if data == 'imagenet':
            self.in_planes = 64
            self.out_planes = 64
            self.num_classes = 1000
            self.conv1 = nn.Conv2d(3, self.out_planes, kernel_size=7, stride=2, padding=3, bias=False)
            self.mask1 = nn.Conv2d(3, self.out_planes, kernel_size=7, stride=2, padding=3, bias=False)
            self.mask1.weight.data = torch.ones(self.mask1.weight.size())
        else:
            self.in_planes = 16
            self.out_planes = 16
            self.num_classes = int(data[5:])
            self.conv1 = nn.Conv2d(3, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.mask1 = nn.Conv2d(3, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.mask1.weight.data = torch.ones(self.mask1.weight.size())

        self.bn1 = nn.BatchNorm2d(self.out_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.out_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.out_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.out_planes * 4, num_blocks[2], stride=2)
        if data == 'imagenet':
            self.layer4 = self._make_layer(block, self.out_planes * 8, num_blocks[3], stride=2)
            self.linear = nn.Linear(self.out_planes*8*block.expansion, self.num_classes)
        else:
            self.linear = nn.Linear(self.out_planes*4*block.expansion, self.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight,  self.mask1.weight)

        out = F.relu(self.bn1(self.conv1(x)))
        if self.num_classes == 1000:
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.num_classes == 1000:
            out = self.layer4(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def __prune__(self, threshold):
        self.mask1.weight.data = torch.gt(torch.abs(self.conv1.weight), threshold).float()
        layers = [self.layer1, self.layer2, self.layer3]
        if self.num_classes == 1000:
            layers.append(self.layer4)
        for layer in layers:
            for sub_block in layer:
                sub_block.__prune__(threshold)
    
cfg_imagenet = {
    '9' : (BasicBlock, [1,1,1,1]),
    '18' : (BasicBlock, [2,2,2,2]),
    '34' : (BasicBlock, [3,4,6,3]),
    '50' : (Bottleneck, [3,4,6,3]),
    '101' : (Bottleneck, [3,4,23,3]),
    '152' : (Bottleneck, [3,8,36,3])
}

cfg_cifar = {
    '20' : (BasicBlock, [3,3,3]),
    '32' : (BasicBlock, [5,5,5]),
    '44' : (BasicBlock, [7,7,7]),
    '56' : (BasicBlock, [9,9,9]),
    '110' : (BasicBlock, [18,18,18]),
    '1202' : (BasicBlock, [200,200,200]),
}

def make_ResNet(depth, data):
    if data == 'imagenet':
        return ResNet(cfg_imagenet[depth][0], cfg_imagenet[depth][1], data)
    if 'cifar' in data:
        return ResNet(cfg_cifar[depth][0], cfg_cifar[depth][1], data)

'''
def ResNet9():
    return ResNet(BasicBlock, [1,1,1,1])

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
'''

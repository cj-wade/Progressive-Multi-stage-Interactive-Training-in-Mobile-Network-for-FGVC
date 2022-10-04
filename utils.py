import random
import torch
from torch.utils.data import DataLoader
import torchvision
import os
from torchvision import transforms as T


def writeLog(path, log):
    f = open(path, 'a+')
    f.write(log)


def saveModel(model, path):
    torch.save(model, path)
    print('save model in {}'.format(path))


def dataTransform(mode='train'):
    print('Default')
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if mode == 'train':
        transforms = T.Compose([
            T.Resize(512),
            T.RandomCrop(448),
            T.ToTensor(),
            normalize
        ])
    else:
        transforms = T.Compose([
            T.Resize(512),
            T.CenterCrop(448),
            T.ToTensor(),
            normalize
        ])

    return transforms


def chooseData(name, batchSize, worker, trainTransforms=None, testTransforms=None):
    trainLoader = None
    testLoader = None
    validLoader = None

    dataTrain = []
    dataTest = []
    dataValid = []

    datasets = os.path.join(os.getcwd(), 'datasets')
    if trainTransforms is None:
        trainTransforms = dataTransform('train')
    if testTransforms is None:
        testTransforms = dataTransform('test')

    # if name == "you_path":
    #     "You can train your own dataset by replacing paths"

    if name == 'CUB':
        # Replace with your path
        root = os.path.join(datasets, 'New_CUB//CUB_200_2011')
        trainRoot = os.path.join(root, 'train')
        testRoot = os.path.join(root, 'test')
        dataTrain = torchvision.datasets.ImageFolder(trainRoot, transform=trainTransforms)
        dataTest = torchvision.datasets.ImageFolder(testRoot, transform=testTransforms)

        trainLoader = torch.utils.data.DataLoader(dataTrain,
                                                  batch_size=batchSize,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=worker)

        testLoader = torch.utils.data.DataLoader(dataTest,
                                                 batch_size=batchSize,
                                                 shuffle=False,
                                                 num_workers=worker)

    if name == "Car":
        # Replace with your path
        root = os.path.join(datasets, './Cars')
        trainRoot = os.path.join(root, 'train')
        testRoot = os.path.join(root, 'test')
        dataTrain = torchvision.datasets.ImageFolder(trainRoot, transform=trainTransforms)
        dataTest = torchvision.datasets.ImageFolder(testRoot, transform=testTransforms)

        trainLoader = torch.utils.data.DataLoader(dataTrain,
                                                  batch_size=batchSize,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=worker)

        testLoader = torch.utils.data.DataLoader(dataTest,
                                                 batch_size=batchSize,
                                                 shuffle=False,
                                                 num_workers=worker)

    if name == "Air":
        # Replace with your path
        root = os.path.join(datasets, './Aircraft')
        trainRoot = os.path.join(root, 'train')
        testRoot = os.path.join(root, 'test')
        dataTrain = torchvision.datasets.ImageFolder(trainRoot, transform=trainTransforms)
        dataTest = torchvision.datasets.ImageFolder(testRoot, transform=testTransforms)

        trainLoader = torch.utils.data.DataLoader(dataTrain,
                                                  batch_size=batchSize,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=worker)

        testLoader = torch.utils.data.DataLoader(dataTest,
                                                 batch_size=batchSize,
                                                 shuffle=False,
                                                 num_workers=worker)

    return trainLoader, testLoader, validLoader, len(dataTrain), len(dataTest), len(dataValid)


def Recursive_Mosaic_Generator(input, r):
    l = []
    for a in range(2):
        for b in range(2):
            l.append([a, b])
    dicing = input.clone()
    block_size = 448
    begin_x = 0
    begin_y = 0
    while (r > 0):
        random.shuffle(l)
        block_size = block_size // 2
        for i in range(4):
            x, y = l[i]
            temp = dicing[..., begin_x:begin_x + block_size, begin_y:begin_y + block_size].clone()
            dicing[..., begin_x:begin_x + block_size, begin_y:begin_y + block_size] = dicing[...,
                                                                                      begin_x + x * block_size: begin_x + (
                                                                                                  x + 1) * block_size,
                                                                                      begin_y + y * block_size:begin_y + (
                                                                                                  y + 1) * block_size].clone()
            dicing[..., begin_x + x * block_size:begin_x + (x + 1) * block_size,
            begin_y + y * block_size:begin_y + (y + 1) * block_size] = temp
        random.shuffle(l)
        x, y = l[i]
        begin_x = begin_x + x * block_size
        begin_y = begin_y + y * block_size
        r = r - 1

    return dicing

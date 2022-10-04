import torch
from torch import nn
from torch.autograd import Variable
import os
from utils import saveModel, chooseData, writeLog, Recursive_Mosaic_Generator as RMG
import time
from models.network.PMSI import PMSI
from models.backbone import MobilenetV2_for_PMSI

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


class Net(nn.Module):
    def __init__(self, model, CLASS=200):
        super(Net, self).__init__()
        for param in model.parameters():
            param.requires_grad = True
        self.pmsi = PMSI(model, feature_size=320, classes_num=CLASS)

    def forward(self, x, train_flag='train', target=None):
        x1, x2, x3, x_concat = self.pmsi(x, train_flag, target)
        return x1, x2, x3, x_concat


def train(modelConfig, dataConfig, logConfig):
    """
    :param modelConfig
    :param dataConfig
    :param logConfig
    :return:
    """
    # modelConfig
    model = modelConfig['model']
    criterion = modelConfig['criterion']
    optimzer_pmsi = modelConfig['optimzer_pmsi']
    optimzer_backbone = modelConfig['optimzer_backbone']
    scheduler_pmsi = modelConfig['scheduler_pmsi']
    scheduler_backbone = modelConfig['scheduler_backbone']
    epochs = modelConfig['epochs']
    device = modelConfig['device']

    # dataConfig
    trainLoader = dataConfig['trainLoader']
    validLoader = dataConfig['validLoader']

    # logConfig
    modelPath = logConfig['modelPath']
    logPath = logConfig['logPath']
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('train is starting in ' + now)

    best_L1_Acc = 0.
    best_L2_Acc = 0.
    best_L3_Acc = 0.
    best_concat_Acc = 0.
    best_mix_Acc = 0.
    best_epoch = 0

    for epoch in range(epochs):
        epoch_log = "Epoch{}/{}".format(epoch, epochs) + "\n" + "-" * 10
        print(epoch_log)

        loss_1, loss_2, loss_3, loss_concat, loss, acc_1, acc_2, acc_3, acc_concat, total_train \
            = oneEpoch_train(model, trainLoader, optimzer_pmsi, optimzer_backbone, scheduler_pmsi, scheduler_backbone,
                             criterion, device, epoch)

        val_loss_1, val_acc_1, val_loss_2, val_acc_2, val_loss_3, val_acc_3, val_loss_concat, val_acc_concat, val_loss_mix, val_acc_mix, total_val \
            = oneEpoch_valid(model, validLoader, criterion, device)

        print("total train:" + str(total_train))
        print("total val:" + str(total_val))

        loss_1 = loss_1 / total_train
        loss_2 = loss_2 / total_train
        loss_3 = loss_3 / total_train
        loss_concat = loss_concat / total_train
        loss = loss / total_train

        acc_1 = acc_1 / total_train
        acc_2 = acc_2 / total_train
        acc_3 = acc_3 / total_train
        acc_concat = acc_concat / total_train

        val_loss_1 = val_loss_1 / total_val
        val_loss_2 = val_loss_2 / total_val
        val_loss_3 = val_loss_3 / total_val
        val_loss_concat = val_loss_concat / total_val
        val_loss_mix = val_loss_mix / total_val

        val_acc_1 = val_acc_1 / total_val
        val_acc_2 = val_acc_2 / total_val
        val_acc_3 = val_acc_3 / total_val
        val_acc_concat = val_acc_concat / total_val
        val_acc_mix = val_acc_mix / total_val

        # save the best model
        if val_acc_1 > best_L1_Acc:
            best_L1_Acc = val_acc_1

        if val_acc_2 > best_L2_Acc:
            best_L2_Acc = val_acc_2

        if val_acc_3 > best_L3_Acc:
            best_L3_Acc = val_acc_3

        if val_acc_concat > best_concat_Acc:
            best_concat_Acc = val_acc_concat

        if val_acc_mix > best_mix_Acc:
            best_epoch = epoch
            best_mix_Acc = val_acc_mix
            saveModel(model, modelPath)

        # training logs
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        train_L1_Log = now + " Train L1 loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(loss_1, 100. * acc_1)
        train_L2_Log = now + " Train L2 loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(loss_2, 100. * acc_2)
        train_L3_Log = now + " Train L3 loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(loss_3, 100. * acc_3)
        train_concat_Log = now + " Train concat loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(loss_concat,
                                                                                                    100. * acc_concat)
        train_total_Log = now + " Train total loss is :{:.4f}\n\n".format(loss)

        val_L1_log = now + " Valid L1 loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(val_loss_1, 100. * val_acc_1)
        val_L2_log = now + " Valid L2 loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(val_loss_2, 100. * val_acc_2)
        val_L3_log = now + " Valid L3 loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(val_loss_3, 100. * val_acc_3)
        val_concat_log = now + " Valid concat loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(val_loss_concat,
                                                                                                  100. * val_acc_concat)
        val_mix_log = now + " Valid mix loss is :{:.4f},Valid accuracy is:{:.4f}%\n\n".format(val_loss_mix,
                                                                                              100. * val_acc_mix)

        best_L1_log = now + ' best L1 Acc is {:.4f}%\n'.format(100. * best_L1_Acc)
        best_L2_log = now + ' best L2 Acc is {:.4f}%\n'.format(100. * best_L2_Acc)
        best_L3_log = now + ' best L3 Acc is {:.4f}%\n'.format(100. * best_L3_Acc)
        best_concat_log = now + ' best concat Acc is {:.4f}%\n'.format(100. * best_concat_Acc)
        best_mix_log = now + ' best mix Acc is {:.4f}%\n'.format(100. * best_mix_Acc)
        best_epoch_log = now + ' best Acc epoch is :' + str(best_epoch) + "\n\n"

        train_log = train_L1_Log + train_L2_Log + train_L3_Log + train_concat_Log + train_total_Log
        val_log = val_L1_log + val_L2_log + val_L3_log + val_concat_log + val_mix_log
        best_log = best_L1_log + best_L2_log + best_L3_log + best_concat_log + best_mix_log + best_epoch_log

        print(train_log + val_log + best_log)
        writeLog(logPath, train_log + val_log + best_log)


def oneEpoch_train(model, dataLoader, optimzer_pmsi, optimzer_backbone, scheduler_pmsi, scheduler_backbone, criterion,
                   device, epoch):
    """
    oneEpoch train
    :param model
    :param dataLoader
    :param criterion
    :return: loss acc total
    """

    model.train()
    loss = 0.
    loss_1 = 0.
    loss_2 = 0.
    loss_3 = 0.
    loss_concat = 0.
    acc_1 = 0.
    acc_2 = 0.
    acc_3 = 0.
    acc_concat = 0.
    total = 0
    # freeze backbone in the first 5 epochs.
    for (inputs, labels) in dataLoader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = Variable(inputs), Variable(labels)
        total += labels.size(0)

        # phase 1
        optimzer_backbone.zero_grad()
        optimzer_pmsi.zero_grad()
        inputs1 = RMG(inputs, 3)
        output_1, _, _, _ = model(x=inputs1, train_flag="train", target=labels)
        _loss_1 = criterion(output_1, labels) * 1
        _loss_1.backward()
        if epoch >= 5:
            optimzer_backbone.step()
        optimzer_pmsi.step()

        # phase 2
        optimzer_backbone.zero_grad()
        optimzer_pmsi.zero_grad()
        inputs2 = RMG(inputs, 2)
        _, output_2, _, _ = model(x=inputs2, train_flag="train", target=labels)
        _loss_2 = criterion(output_2, labels) * 1
        _loss_2.backward()
        if epoch >= 5:
            optimzer_backbone.step()
        optimzer_pmsi.step()

        # phase 3
        optimzer_backbone.zero_grad()
        optimzer_pmsi.zero_grad()
        inputs3 = RMG(inputs, 1)
        _, _, output_3, _ = model(x=inputs3, train_flag="train", target=labels)
        _loss_3 = criterion(output_3, labels) * 1
        _loss_3.backward()
        if epoch >= 5:
            optimzer_backbone.step()
        optimzer_pmsi.step()

        # phase 4
        optimzer_backbone.zero_grad()
        optimzer_pmsi.zero_grad()
        _, _, _, output_4 = model(x=inputs, train_flag="train", target=labels)
        _loss_concat = criterion(output_4, labels) * 2
        _loss_concat.backward()
        if epoch >= 5:
            optimzer_backbone.step()
        optimzer_pmsi.step()

        scheduler_backbone.step()
        scheduler_pmsi.step()

        _, preds_1 = torch.max(output_1.data, 1)
        _, preds_2 = torch.max(output_2.data, 1)
        _, preds_3 = torch.max(output_3.data, 1)
        _, preds = torch.max(output_4.data, 1)

        loss += (_loss_1.item() + _loss_2.item() + _loss_3.item() + _loss_concat.item())
        loss_1 += _loss_1.item()
        loss_2 += _loss_2.item()
        loss_3 += _loss_3.item()
        loss_concat += _loss_concat.item()

        acc_1 += torch.sum(preds_1 == labels).item()
        acc_2 += torch.sum(preds_2 == labels).item()
        acc_3 += torch.sum(preds_3 == labels).item()
        acc_concat += torch.sum(preds == labels).item()

    return loss_1, loss_2, loss_3, loss_concat, loss, acc_1, acc_2, acc_3, acc_concat, total


def oneEpoch_valid(model, dataLoader, criterion, device):
    """
    oneEpoch valid
    :param model
    :param dataLoader
    :param criterion
    :return: loss acc total
    """
    with torch.no_grad():
        model.eval()
        loss_1 = 0.
        loss_2 = 0.
        loss_3 = 0.
        loss_concat = 0.
        loss_mix = 0.
        acc_1 = 0.
        acc_2 = 0.
        acc_3 = 0.
        acc_concat = 0.
        acc_mix = 0.
        total = 0
        for (inputs, labels) in dataLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)
            total += labels.size(0)

            outputs1, outputs2, outputs3, outputs_concat = model(x=inputs, train_flag="val")

            outputs_mix = outputs1 + outputs2 + outputs3 + outputs_concat

            _loss_1 = criterion(outputs1, labels)
            _loss_2 = criterion(outputs2, labels)
            _loss_3 = criterion(outputs3, labels)
            _loss_concat = criterion(outputs_concat, labels)
            _loss_mix = criterion(outputs_mix, labels)

            _, preds_1 = torch.max(outputs1.data, 1)
            _, preds_2 = torch.max(outputs2.data, 1)
            _, preds_3 = torch.max(outputs3.data, 1)
            _, preds_concat = torch.max(outputs_concat.data, 1)
            _, predicted_mix = torch.max(outputs_mix.data, 1)

            loss_1 += _loss_1.item()
            loss_2 += _loss_2.item()
            loss_3 += _loss_3.item()
            loss_concat += _loss_concat.item()
            loss_mix += _loss_mix.item()

            acc_1 += torch.sum(preds_1 == labels).item()
            acc_2 += torch.sum(preds_2 == labels).item()
            acc_3 += torch.sum(preds_3 == labels).item()
            acc_concat += torch.sum(preds_concat == labels).item()
            acc_mix += torch.sum(predicted_mix == labels).item()

    return loss_1, acc_1, loss_2, acc_2, loss_3, acc_3, loss_concat, acc_concat, loss_mix, acc_mix, total


def _train(lr=3e-4, weight_decay=5e-4, momentum=0.9):
    # torch.backends.cudnn.benchmark = True
    class_num = 200
    print("cuda:0, 1")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(MobilenetV2_for_PMSI.mobilenet_v2(pretrained=True), class_num)
    device_ids = [0, 1]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    optimzer_backbone = torch.optim.SGD([
        {'params': model.module.pmsi.features.parameters(), 'lr': lr * 1.},
    ],
        momentum=momentum, weight_decay=weight_decay)

    optimzer_pmsi = torch.optim.SGD([
        {'params': model.module.pmsi.map1.parameters(), 'lr': lr * 10.},
        {'params': model.module.pmsi.map2.parameters(), 'lr': lr * 10.},
        {'params': model.module.pmsi.smooth_conv1.parameters(), 'lr': lr * 10.},
        {'params': model.module.pmsi.classifier1.parameters(), 'lr': lr * 10.},
        {'params': model.module.pmsi.smooth_conv2.parameters(), 'lr': lr * 10.},
        {'params': model.module.pmsi.classifier2.parameters(), 'lr': lr * 10.},
        {'params': model.module.pmsi.smooth_conv3.parameters(), 'lr': lr * 10.},
        {'params': model.module.pmsi.classifier3.parameters(), 'lr': lr * 10.},
        {'params': model.module.pmsi.classifier_concat.parameters(), 'lr': lr * 10.},
    ],
        momentum=momentum, weight_decay=weight_decay)

    epochs = 150
    batchSize = 32
    worker = 8

    from torchvision import transforms as T
    # 自定义数据增强方式
    trainTransforms = T.Compose([
        T.Resize((512, 512)),
        T.RandomCrop(448),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testTransforms = T.Compose([
        T.Resize((512, 512)),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('CUB', batchSize, worker,
                                                                                            trainTransforms,
                                                                                            testTransforms)
    print("train_length:" + str(trainLength))
    print("test_length:" + str(testLength))

    scheduler_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer_backbone, T_max=150 * trainLength)
    scheduler_pmsi = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer_pmsi, T_max=150 * trainLength)
    modelConfig = {
        'model': model,
        'criterion': criterion,
        'optimzer_backbone': optimzer_backbone,
        'optimzer_pmsi': optimzer_pmsi,
        'scheduler_backbone': scheduler_backbone,
        'scheduler_pmsi': scheduler_pmsi,
        'epochs': epochs,
        'device': device
    }

    dataConfig = {
        'trainLoader': trainLoader,
        'validLoader': testLoader
    }

    modelPath = os.path.join(os.getcwd(), 'checkpoints', 'RMG_PMSI.pth')
    logPath = os.path.join(os.getcwd(), 'logs', 'train_log.txt')

    logConfig = {
        'modelPath': modelPath,
        'logPath': logPath
    }

    train(modelConfig, dataConfig, logConfig)


if __name__ == '__main__':
    print(torch.__version__)
    _train(lr=3e-4, weight_decay=5e-4, momentum=0.9)

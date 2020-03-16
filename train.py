import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import time
import numpy as np
import matplotlib.pyplot as plt
import models
import config
import os

train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),#随机旋转
        transforms.RandomHorizontalFlip(),#随机水平翻转
        transforms.CenterCrop(size=128),#中心裁剪到224*224
        transforms.ToTensor(),#转化成张量
        transforms.Normalize([0.485, 0.456, 0.406],#归一化
                             [0.229, 0.224, 0.225])
])

test_valid_transforms = transforms.Compose(
        [transforms.Resize(128),
         transforms.CenterCrop(128),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
         [0.229, 0.224, 0.225])])


train_directory = config.TRAIN_DATASET_DIR
valid_directory = config.VALID_DATASET_DIR

batch_size = config.BATCH_SIZE
num_classes = config.NUM_CLASSES
lr=config.LR
resume=config.PRETRAINED_MODEL



train_datasets = datasets.ImageFolder(train_directory, transform=train_transforms)
train_data_size = len(train_datasets)
train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

valid_datasets = datasets.ImageFolder(valid_directory,transform=test_valid_transforms)
valid_data_size = len(valid_datasets)
valid_data = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

print(train_data_size, valid_data_size)


def train_and_valid(model, loss_function, optimizer, scheduler,epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#若有gpu可用则用gpu
    record = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):#训练epochs轮
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        scheduler.step()#每个epoch开始时都要更新学习率
        model.train()#设置模型状态为训练状态

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(labels)
            # 记得清零，将网络中的所有梯度置0
            optimizer.zero_grad()
            #网络的前向传播
            outputs = model(inputs)
            #计算损失，将输出的outputs和原来导入的labels作为loss函数的输入
            loss = loss_function(outputs, labels)
            #反向传播
            loss.backward()
            #更新参数（梯度）
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()#验证

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if avg_valid_acc > best_acc  :#记录最高准确性的模型
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        # if (epoch+1) % config.save_epoch_freq == 0:
        #     if not os.path.exists(config.save_path):
        #         os.makedirs(config.save_path)
        #     torch.save(model, os.path.join(config.save_path, "epoch_" + str(epoch) + ".pth"))

    return model, record


if __name__=='__main__':
    num_epochs = config.NUM_EPOCHS
    resnext101 = models.resnext101_32x8d(pretrained=True)

    freeze_conv_layer = False
    """
    if freeze_conv_layer:
        for param in resnext101.parameters():  # freeze layers
            param.requires_grad = False
    num_ftrs = resnext101.fc.in_features
    resnext101.fc = nn.Linear(num_ftrs, num_classes)
    """
    print('before:{%s}\n' % resnext101)
    if freeze_conv_layer:
        for param in resnext101.parameters():
            param.requires_grad = False
    fc_inputs = resnext101.fc.in_features
    resnext101.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )
    print('after:{%s}\n' % resnext101)


    """
    if resume:
        if os.path.isfile(resume):
            print(("=> loading checkpoint '{}'".format(resume)))
            checkpoint = torch.load(resume)
            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
            resnext101.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(resume)))
    """
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnext101.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    #学习率的变化策略，每隔step_size个epoch就将学习率降为原来的gamma倍
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    trained_model, record = train_and_valid(resnext101, loss_func, optimizer, exp_lr_scheduler,num_epochs)
    torch.save(trained_model, config.TRAINED_MODEL)

    record = np.array(record)
    plt.plot(record[:, 0:2])
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig('loss.png')
    plt.show()

    plt.plot(record[:, 2:4])
    plt.legend(['Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('accuracy.png')
    plt.show()

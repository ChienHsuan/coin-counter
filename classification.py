import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from cls_dataset import CoinDataset


class Train(object):
    def __init__(self, class_names, model, train_loader, valid_loader, test_loader, epochs, init_lr, valid_step=1):
        self.train_loss_history = []
        self.train_acc_history = []
        self.valid_acc_history = []
        
        self.classes_name = class_names
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.epochs = epochs
        self.valid_step = valid_step
        
        self.model = model
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs-1, eta_min=5e-6)
        
        self._run()
        print('\n======================================================\n')
        self._test()
        
    def _calculate_acc(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).sum().item() / labels.shape[0]
        return acc
    
    def _valid_evaluation(self):
        self.model.eval()
        with torch.no_grad():
            total_acc = 0
            for data in self.valid_loader:
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()

                outputs = self.model(imgs)
                acc = self._calculate_acc(outputs.data, labels)
                total_acc += acc

            return total_acc / len(self.valid_loader)
    
    def _test(self):
        self.model.eval()
        with torch.no_grad():
            # total accuracy and class accuracy
            class_correct = [0. for i in range(len(self.classes_name))]
            class_total = [0. for i in range(len(self.classes_name))]

            for data in self.test_loader:
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()

                outputs = self.model(imgs)

                _, predicted = torch.max(outputs.data, 1)
                batch_correct_class = (predicted == labels)
                for i in range(labels.shape[0]):
                    label = labels[i]
                    class_correct[label] += batch_correct_class[i].item()
                    class_total[label] += 1
            for i in range(len(self.classes_name)):
                print(f'Accuracy of {self.classes_name[i]}: {100 * class_correct[i] / class_total[i]} %')
            print('')
            print(f'Accuracy of the network on all testset: {100 * sum(class_correct) / sum(class_total)} %')
    
    def plot_loss(self):
        fig = plt.figure(figsize=(12,4))
        ax = fig.gca()
        plt.plot(list(range(1, self.epochs+1)), self.train_loss_history)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        plt.title('Training loss')
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.grid(color='r', linestyle='dotted', linewidth=1)
        plt.xlim([1, self.epochs])
        plt.show()
        
    def plot_acc(self):
        fig = plt.figure(figsize=(12,4))
        ax = fig.gca()
        acc_1, = plt.plot(list(range(1, self.epochs+1)), self.train_acc_history, '-b')
        acc_2, = plt.plot(list(range(1, self.epochs+1)), self.valid_acc_history, '-g')
        ax.xaxis.set_major_locator(MultipleLocator(1))
        plt.title('Training and validation accuracy')
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.legend([acc_1, acc_2], ['training accuracy', 'validation accuracy'], loc='lower right')
        plt.grid(color='r', linestyle='dotted', linewidth=1)
        plt.xlim([1, self.epochs])
        plt.show()
    
    def _run(self):
        for epoch in range(self.epochs):
            total_loss = 0
            total_acc = 0

            self.model.train()
            for iteration, data in enumerate(self.train_loader):
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()

                outputs = self.model(imgs)

                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                acc = self._calculate_acc(outputs.data, labels)
                total_acc += acc
                if (iteration % 100) == 0:
                    print(f"Epoch[{epoch}][{iteration}/{len(self.train_loader)-1}]:    loss: {loss.item()}    acc: {acc*100} .")        

            total_loss /= len(self.train_loader)
            self.train_loss_history.append(total_loss)
            total_acc = (total_acc/len(self.train_loader))*100
            self.train_acc_history.append(total_acc)
            print(f'Epoch[{epoch}]:    loss: {total_loss}    acc: {total_acc} .')

            self.lr_scheduler.step()

            if epoch % self.valid_step == 0 or epoch == self.epochs-1:
                valid_acc = self._valid_evaluation()
                self.valid_acc_history.append(valid_acc*100)


def plot_acc(epochs, curve_1, curve_2, title):
    fig = plt.figure(figsize=(12,4))
    ax = fig.gca()
    acc_1, = plt.plot(list(range(1, epochs+1)), curve_1, '-b')
    acc_2, = plt.plot(list(range(1, epochs+1)), curve_2, '-g')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend([acc_1, acc_2], ['resnet50', 'mobilenetv2'], loc='lower right', framealpha=0.5)
    plt.grid(color='r', linestyle='dotted', linewidth=1)
    plt.xlim([1, epochs])
    plt.show()


def main():
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = CoinDataset('/USER-DEFINED-PATH/coins/Train/', transform=train_transform)
    print(f'train_num_imgs: {len(trainset)}')
    validationset = CoinDataset('/USER-DEFINED-PATH/coins/Validation/', transform=valid_transform)
    print(f'validation_num_imgs: {len(validationset)}')
    testset = CoinDataset('/USER-DEFINED-PATH/coins/Test/', transform=test_transform)
    print(f'test_num_imgs: {len(testset)}')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(validationset, batch_size=32, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, num_workers=4, pin_memory=True)

    # training setting
    epochs = 30
    init_lr = 0.0003
    valid_step = 1

    # resnet 50
    resnet50_model = models.resnet50(pretrained=False, num_classes=len(trainset.class_names))
    resnet50_model_dict = resnet50_model.state_dict()
    resnet50_pretrained_weight = torch.load('pre-trained_models/resnet50-19c8e357.pth')
    resnet50_pretrained_weight = {k:v for k, v in resnet50_pretrained_weight.items() if (k in resnet50_model_dict and 'fc' not in k)}
    resnet50_model.load_state_dict(resnet50_pretrained_weight, strict=False)
    resnet50_model = resnet50_model.cuda()
    resnet50_model_train = Train(trainset.class_names, resnet50_model, train_loader, valid_loader, test_loader, epochs, init_lr, valid_step=valid_step)

    resnet50_model_train.plot_loss()
    plt.savefig('resnet50_1.png')
    resnet50_model_train.plot_acc()
    plt.savefig('resnet50_2.png')

    # mobilenet v2
    mobilenetv2_model = models.mobilenet_v2(pretrained=False, num_classes=len(trainset.class_names))
    mobilenetv2_model_dict = mobilenetv2_model.state_dict()
    mobilenetv2_pretrained_weight = torch.load('pre-trained_models/mobilenet_v2-b0353104.pth')
    mobilenetv2_pretrained_weight = {k:v for k, v in mobilenetv2_pretrained_weight.items() if (k in mobilenetv2_model_dict and 'classifier' not in k)}
    mobilenetv2_model.load_state_dict(mobilenetv2_pretrained_weight, strict=False)
    mobilenetv2_model = mobilenetv2_model.cuda()
    mobilenetv2_model_train = Train(trainset.class_names, mobilenetv2_model, train_loader, valid_loader, test_loader, epochs, init_lr, valid_step=valid_step)

    mobilenetv2_model_train.plot_loss()
    plt.savefig('mobilenetv2_1.png')
    mobilenetv2_model_train.plot_acc()
    plt.savefig('mobilenetv2_2.png')

    # save model
    torch.save(resnet50_model.state_dict(), 'trained-models/resnet50.pth')
    torch.save(mobilenetv2_model.state_dict(), 'trained-models/mobilenetv2.pth')

    # comparison
    plot_acc(epochs, resnet50_model_train.train_loss_history, mobilenetv2_model_train.train_loss_history, 'Training loss')
    plt.savefig('training_loss.png')
    plot_acc(epochs, resnet50_model_train.train_acc_history, mobilenetv2_model_train.train_acc_history, 'Training accuracy')
    plt.savefig('training_acc.png')
    plot_acc(epochs, resnet50_model_train.valid_acc_history, mobilenetv2_model_train.valid_acc_history, 'Validation accuracy')
    plt.savefig('validation_acc.png')

if __name__ == '__main__':
    main()

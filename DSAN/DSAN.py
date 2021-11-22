from __future__ import print_function
import torch
import torch.nn.functional as F
import os
import math
from utils import data_loader
from utils.config import *
import time
import argparse

from net import ResNet as models

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser(description='DSAN Model Training')
# params necessary
parser.add_argument('--target_model', type=str, default=None, metavar='M',
                    help='target model access')

parser.add_argument('--source_name', type=str, default=None, metavar='S',
                    help='source data')

parser.add_argument('--target_name', type=str, default=None, metavar='T',
                    help='target data')

args = parser.parse_args()

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source_loader = data_loader.load_training_data(root_path, os.path.join(args.source_name, 'train'), batch_size, kwargs)
# change training data --> testing data
target_train_loader = data_loader.load_training_data(root_path, os.path.join(args.target_name, 'test'), batch_size, kwargs)
target_test_loader = data_loader.load_testing_data(root_path, os.path.join(args.target_name, 'test'), batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)


def train(epoch, model, t_model):

    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE) )

    if bottle_neck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader

    for i in range(1, num_iter):

        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()

        optimizer.zero_grad()

        label_source_pred, loss_mmd = model(data_source, data_target, label_source, t_model)

        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)

        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        loss = loss_cls + param * lambd * loss_mmd

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                epoch, epochs, i * len(data_source), len_source_dataset,
                100. * i / len_source_loader, loss.item(), loss_cls.item(), loss_mmd.item()))


def test(model):

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            s_output, t_output = model(data, data, target)

            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target).item() # sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability

            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len_target_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            args.target_name, test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))

    return correct


if __name__ == '__main__':

    model = models.DSAN(num_classes=class_num)

    # access target model
    target_model = torch.load(args.target_model)

    correct = 0
    # print(model)
    if cuda:
        model.cuda()
    time_start = time.time()

    for epoch in range(1, epochs + 1):
        # model training
        train(epoch, model, target_model)

        # model testing
        t_correct = test(model)

        if t_correct > correct:
            correct = t_correct
            # model saved
            torch.save(model.state_dict(), 'model_resnet50_{}_{}.pth'.format(args.source_name, args.target_name))

        end_time = time.time()
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
              args.source_name, args.target_name, correct, 100. * correct / len_target_dataset ))
        print('cost time:', end_time - time_start)

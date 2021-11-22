from torchvision import models
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os

import tqdm
import copy

from torchvision import transforms, datasets


parser = argparse.ArgumentParser(description='Target Model Training')
# params necessary
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--num_classes', type=int, default=65, metavar='N',
                    help='number of classes for training (default: 10)')

parser.add_argument('--model_name', type=str, default='resnet',
                    help='model name of the target')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')

parser.add_argument('--data_path', type=str, default='../../datasets/Office31/webcam/',
                    help='the path to load the data')

parser.add_argument('--cate_name', type=str, default='train',
                    help='the name of category')

parser.add_argument('--save_path', type=str, default='./',
                    help='the path to save the target model')

parser.add_argument('--seed', type=int, default=233, metavar='S',
                    help='random seed (default: 1)')
                    
parser.add_argument('--gpu', default=3, type=int)

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def load_data(input_size):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_path, x), data_transforms[x])
                      for x in ['train', 'test']}
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, **kwargs)
                    for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    return data_loaders, dataset_sizes


def model_training(model, data_loaders, dataset_sizes):
    """
    training target attack model
    :param epoch: num of epoch
    :param model: target model architecture
    :param train_loader: training data loader
    :return:
    """

    best_acc = 0.0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            print('{0} Phase: {1} {2}'.format('-' * 10, phase, '-' * 10))

            for batch_idx, (inputs, labels, _) in tqdm.tqdm(enumerate(data_loaders[phase]),
                                                         total=int(dataset_sizes[phase]/args.batch_size),
                                                         desc='Train epoch = {}'.format(epoch),
                                                         ncols=80, leave=False):

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(preds)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        # scheduler.step()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, 100 * epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:

                best_acc = epoch_acc
                print('Best val Acc: {:2f}%'.format(100 * best_acc))

                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)  # success or not
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                model_saved_path = os.path.join(args.save_path, 'target_{}_'.format(args.cate_name) + args.model_name + '.pt')
                torch.save(model, model_saved_path)

        print()

    return model, model_saved_path


def model_test(model_saved_path, data_loaders):
    print()
    print('starting offline testing')
    print()

    model = torch.load(model_saved_path)

    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels, _) in data_loaders['test']:

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).argmax(axis=-1)
            # print(outputs)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()

    print('Accuracy of the network on the test images: {:2f}%'.format(100 * correct / total))


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    Initialize these variables which will be set in this if statement. Each of these
    variables is model specific.'''
    :param model_name:
    :param num_classes:
    :param feature_extract:
    :param use_pretrained:
    :return:
    """

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg16":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg19":
        """ VGG19
        """
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


if __name__ == '__main__':

    # Initialize the target model for this run
    model, input_size = initialize_model(args.model_name, args.num_classes, feature_extract=True, use_pretrained=True)
    model.to(DEVICE)
    # Print the model we just instantiated
    print(model)

    data_loaders, dataset_sizes = load_data(input_size)
    print(dataset_sizes)

    model, model_saved_path = model_training(model, data_loaders, dataset_sizes)
    print('Finished Training')
    
    # test offline
    model_test(model_saved_path, data_loaders)


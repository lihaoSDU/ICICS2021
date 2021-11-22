from torchvision import datasets, transforms
import torch
import os


def load_data(data_folder, batch_size, train_flag, kwargs):
    """load image data by leveraging torchvision.datasets.ImageFolder from data folders."""
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train_flag else 'test'])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs,
                                              drop_last=True if train_flag else False)

    dataset_sizes = len(data)

    return data_loader, data.class_to_idx, dataset_sizes


def load_training_data(root_path, dir, batch_size, kwargs):

    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_testing_data(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader


def load_data_with_transform(data_folder, batch_size, train_flag, kwargs):
    """
    Load image data by leveraging torchvision.datasets.ImageFolder from data folders.
    This function added several image transform tricks for fuzzing.
    """

    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
             transforms.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.5, hue=0),
             # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 0.9), shear=1, fillcolor=(50, 1, 1)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])],
             transforms.RandomErasing(ratio=(0.1, 0.2), value='random'),
        )
    }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train_flag else 'test'])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs,
                                              drop_last=True if train_flag else False)

    dataset_sizes = len(data)

    return data_loader, data.class_to_idx, dataset_sizes

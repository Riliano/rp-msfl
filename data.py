import os
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from arguments import Arguments


def load_data(dataset):
    data_loc = './storage'
    train = None
    test = None

    if dataset == 'cifar10':
        train_transform = transforms.Compose([
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224), # preprocessing - need to change the model for this
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train = datasets.CIFAR10(root=data_loc, train=True,  download=True, transform=train_transform)
        test  = datasets.CIFAR10(root=data_loc, train=False, download=True, transform=test_transform)
    elif dataset == 'fashionmnist':
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = datasets.FashionMNIST(root=data_loc, train=True, download=True, transform=train_transform)
        test  = datasets.FashionMNIST(root=data_loc, train=False, download=True, transform=train_transform)

    testloader = torch.utils.data.DataLoader(test, batch_size=len(test))
    testsample, testlabel = next(iter(testloader))

    print('total data train len: ', len(train))
    print('total data test  len: ', len(test))

    return train, [testsample, testlabel]

def tensor_loader(args):
    shuffle_file = './storage/' + args.dataset + '_shuffle.pkl'
    data = pickle.load(open('./storage/' + args.dataset + '_data_ind.pkl', 'rb'))
    train = data[0]
    test = data[1]

    traindata_split = torch.utils.data.random_split(train, ([train.data.shape[0] // args.clients] * args.clients))
    train_loaders = [torch.utils.data.DataLoader(x, batch_size=args.batch_size, shuffle=True) for x in traindata_split]

    print('total data len: ', len(train) + len(test[0]))
    print('total tr len %d | val len %d' % (len(train), len(test[0])))
    for i in range(args.clients):
        print('user %d tr len %d' % (i, len(train_loaders[i].dataset)))

    return train_loaders, test


def main():
    args = Arguments()
    for dataset in args.available_datasets:
        print('Getting dataset: ' + dataset)
        train, validate = load_data(dataset)
        pickle.dump([train, validate], open('./storage/' + dataset + '_data_ind.pkl', 'wb'))
        print('Success')
    print('Done')


if __name__ == "__main__":
    main()

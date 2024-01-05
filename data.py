import os
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from arguments import Arguments


def load_data(dataset):
    shuffle_file = './storage/' + dataset + '_shuffle.pkl'
    data_loc = './storage'
    # load the train dataset
    train = []
    test = []

    if dataset == 'cifar10':
        train_transform = transforms.Compose([
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224), # alexnet preprocessing - expensive
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=train_transform)
        test  = datasets.CIFAR10(root=data_loc, train=False, download=True, transform=train_transform)
    elif dataset == 'fashionmnist':
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = datasets.FashionMNIST(root=data_loc, train=True, download=True, transform=train_transform)
        test  = datasets.FashionMNIST(root=data_loc, train=False, download=True, transform=train_transform)

### !!!! #####
#    traindata_split = torch.utils.data.random_split(train, \
#                                                    [int(train.data.shape[0] / 10) \
#                                                    for _ in range(10)])
#
    testloader = torch.utils.data.DataLoader(test, batch_size=len(test))
    testsample, testlabel = next(iter(testloader))

#    print("Hi")
#    print(train.data)
#    print(traindata_split)
#    print(len(traindata_split))
#    train_loader = [torch.utils.data.DataLoader(x, batch_size=256, shuffle=True) for x in traindata_split]
#    print("Blq")
#    print(train_loader)
#    print(len(train_loader))
#    print(train_loader[0])
#    throwup = iter(train_loader[0])
#    for sample, label in testloader:
#        print('dt')
#        print(type(sample))
#        print(sample.shape)
#        print(type(label))
#        print(label.shape)
#        print(label)
#    print('wut')

#    X = []
#    Y = []
#    print("I'm here btw")
#    for i in range(len(train)):
#        X.append(train[i][0].numpy())
#        Y.append(train[i][1])
#
#    for i in range(len(test)):
#        X.append(test[i][0].numpy())
#        Y.append(test[i][1])
#
#    X = np.array(X)
#    Y = np.array(Y)

    print('total data train len: ', len(train))
    print('total data test  len: ', len(test))

#    if not os.path.isfile(shuffle_file):
#        all_indices = np.arange(len(X))
#        np.random.shuffle(all_indices)
#        pickle.dump(all_indices, open(shuffle_file, 'wb'))
#    else:
#        all_indices = pickle.load(open(shuffle_file, 'rb'))
#
#    X = X[all_indices]
#    Y = Y[all_indices]

#    return X, Y
    return train, [testsample, testlabel]

class TrainingTensors:
    def __init__(self, user_tr_data, user_tr_label, val_data, val_label, te_data, te_label):
        self.user_tr_data  = user_tr_data
        self.user_tr_label = user_tr_label
        self.val_data      = val_data
        self.val_label     = val_label
        self.te_data       = te_data
        self.te_label      = te_label

def tensor_loader(args):
    shuffle_file = './storage/' + args.dataset + '_shuffle.pkl'
    data = pickle.load(open('./storage/' + args.dataset + '_data_ind.pkl', 'rb'))
#    X = data[0]
#    Y = data[1]
    # data loading
    train = data[0]
    test = data[1]

    traindata_split = torch.utils.data.random_split(train, ([train.data.shape[0] // args.clients] * args.clients))
#                                                    [int(train.data.shape[0] // args.clients) \
#                                                    for _ in range(args.clients)])

    print(len(traindata_split))

    train_loaders = [torch.utils.data.DataLoader(x, batch_size=args.batch_size, shuffle=True) for x in traindata_split]

    print('total data len: ', len(train) + len(test[0]))
    print(len(train_loaders))
    print(type(train_loaders[0]))
    print(len(train_loaders[0]))
    print(len(train_loaders[0].dataset))

#    if not os.path.isfile(shuffle_file):
#        all_indices = np.arange(len(X))
#        np.random.shuffle(all_indices)
#        pickle.dump(all_indices, open(shuffle_file, 'wb'))
#    else:
#        all_indices = pickle.load(open(shuffle_file, 'rb'))

    #total_tr_data = X[:args.total_tr_len]
    #total_tr_label = Y[:args.total_tr_len]

    #val_data = X[args.total_tr_len:(args.total_tr_len + args.val_len)]
    #val_label = Y[args.total_tr_len:(args.total_tr_len + args.val_len)]

    #te_data = X[(args.total_tr_len + args.val_len):(args.total_tr_len + args.val_len + args.te_len)]
    #te_label = Y[(args.total_tr_len + args.val_len):(args.total_tr_len + args.val_len + args.te_len)]

    #total_tr_data_tensor = torch.from_numpy(total_tr_data).type(torch.FloatTensor)
    #total_tr_label_tensor = torch.from_numpy(total_tr_label).type(torch.LongTensor)

    #val_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor)
    #val_label_tensor = torch.from_numpy(val_label).type(torch.LongTensor)

    #te_data_tensor = torch.from_numpy(te_data).type(torch.FloatTensor)
    #te_label_tensor = torch.from_numpy(te_label).type(torch.LongTensor)

    print('total tr len %d | val len %d' % (
        len(train), len(test[0])))

#
#    user_tr_data_tensors = []
#    user_tr_label_tensors = []
#
    for i in range(args.clients):
#        user_tr_data_tensor = torch.from_numpy(total_tr_data[args.user_tr_len * i:args.user_tr_len * (i + 1)]).type(torch.FloatTensor)
#        user_tr_label_tensor = torch.from_numpy(total_tr_label[args.user_tr_len * i:args.user_tr_len * (i + 1)]).type(
#            torch.LongTensor)
#
#        user_tr_data_tensors.append(user_tr_data_tensor)
#        user_tr_label_tensors.append(user_tr_label_tensor)
        print('user %d tr len %d' % (i, len(train_loaders[i].dataset)))
#    print('test')
#    hi = iter(train_loaders[0])
#
#    print('blq')
#    print(enumerate(train_loaders[0]))
#    for i in enumerate(train_loaders[0]):
#        print('hi')
#        print(i)
#        d, l = train_loaders[0][i]
#        print(d.shape, l.shape)
#        print(l)
#    print('end')
#
#    return TrainingTensors(user_tr_data_tensors, \
#                          user_tr_label_tensors, \
#                          val_data_tensor, \
#                          val_label_tensor, \
#                          te_data_tensor, \
#                          te_label_tensor)
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

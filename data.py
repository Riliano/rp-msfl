import os
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from arguments import Arguments


def load_data(args):
    data_loc = './utils'
    # load the train dataset

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar10_train = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=train_transform)

    cifar10_test = datasets.CIFAR10(root=data_loc, train=False, download=True, transform=train_transform)

    X = []
    Y = []
    for i in range(len(cifar10_train)):
        X.append(cifar10_train[i][0].numpy())
        Y.append(cifar10_train[i][1])

    for i in range(len(cifar10_test)):
        X.append(cifar10_test[i][0].numpy())
        Y.append(cifar10_test[i][1])

    X = np.array(X)
    Y = np.array(Y)

    print('total data len: ', len(X))

    if not os.path.isfile('./cifar10_shuffle.pkl'):
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices, open('./cifar10_shuffle.pkl', 'wb'))
    else:
        all_indices = pickle.load(open('./cifar10_shuffle.pkl', 'rb'))

    X = X[all_indices]
    Y = Y[all_indices]

    return X, Y

class TrainingTensors:
    def __init__(self, user_tr_data, user_tr_label, val_data, val_label, te_data, te_label):
        self.user_tr_data  = user_tr_data
        self.user_tr_label = user_tr_label
        self.val_data      = val_data
        self.val_label     = val_label
        self.te_data       = te_data
        self.te_label      = te_label

def tensor_loader(args):

    data = pickle.load(open('./cifar10_data_ind.pkl', 'rb'))
    X = data[0]
    Y = data[1]
    # data loading

    print('total data len: ', len(X))

    if not os.path.isfile('./cifar10_shuffle.pkl'):
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices, open('./cifar10_shuffle.pkl', 'wb'))
    else:
        all_indices = pickle.load(open('./cifar10_shuffle.pkl', 'rb'))

    total_tr_data = X[:args.total_tr_len]
    total_tr_label = Y[:args.total_tr_len]

    val_data = X[args.total_tr_len:(args.total_tr_len + args.val_len)]
    val_label = Y[args.total_tr_len:(args.total_tr_len + args.val_len)]

    te_data = X[(args.total_tr_len + args.val_len):(args.total_tr_len + args.val_len + args.te_len)]
    te_label = Y[(args.total_tr_len + args.val_len):(args.total_tr_len + args.val_len + args.te_len)]

    total_tr_data_tensor = torch.from_numpy(total_tr_data).type(torch.FloatTensor)
    total_tr_label_tensor = torch.from_numpy(total_tr_label).type(torch.LongTensor)

    val_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor)
    val_label_tensor = torch.from_numpy(val_label).type(torch.LongTensor)

    te_data_tensor = torch.from_numpy(te_data).type(torch.FloatTensor)
    te_label_tensor = torch.from_numpy(te_label).type(torch.LongTensor)

    print('total tr len %d | val len %d | test len %d' % (
        len(total_tr_data_tensor), len(val_data_tensor), len(te_data_tensor)))


    user_tr_data_tensors = []
    user_tr_label_tensors = []

    for i in range(args.clients):
        user_tr_data_tensor = torch.from_numpy(total_tr_data[args.user_tr_len * i:args.user_tr_len * (i + 1)]).type(torch.FloatTensor)
        user_tr_label_tensor = torch.from_numpy(total_tr_label[args.user_tr_len * i:args.user_tr_len * (i + 1)]).type(
            torch.LongTensor)

        user_tr_data_tensors.append(user_tr_data_tensor)
        user_tr_label_tensors.append(user_tr_label_tensor)
        print('user %d tr len %d' % (i, len(user_tr_data_tensor)))

    return TrainingTensors(user_tr_data_tensors, \
                          user_tr_label_tensors, \
                          val_data_tensor, \
                          val_label_tensor, \
                          te_data_tensor, \
                          te_label_tensor)


def main():
    print("Hello")
    X, Y = load_data(Arguments())
    pickle.dump([X, Y], open('./cifar10_data_ind.pkl', 'wb'))


if __name__ == "__main__":
    main()

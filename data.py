import os
import numpy as np
import pickle
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


def main():
    print("Hello")
    X, Y = load_data(Arguments())
    pickle.dump([X, Y], open('./cifar10_data_ind.pkl', 'wb'))


if __name__ == "__main__":
    main()

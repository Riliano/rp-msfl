from __future__ import print_function

import os
import pickle
import csv

import numpy as np

import torch.nn as nn

from aggregations import fedmes_median, fedmes_mean
from cifar10.cifar10_normal_train import *

from cifar10.cifar10_models import *

from cifar10.sgd import SGD

from arguments import Arguments

from attack import min_max_attack, lie_attack, get_malicious_updates_fang_trmean
from client import Client

args = Arguments()

# X, Y = load_data(args)
data = pickle.load(open('./cifar10_data_ind.pkl', 'rb'))
X = data[0]
Y = data[1]
# data loading

nusers = args.clients
user_tr_len = 2400

total_tr_len = user_tr_len * nusers
val_len = 3300
te_len = 3300

print('total data len: ', len(X))

if not os.path.isfile('./cifar10_shuffle.pkl'):
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices, open('./cifar10_shuffle.pkl', 'wb'))
else:
    all_indices = pickle.load(open('./cifar10_shuffle.pkl', 'rb'))

total_tr_data = X[:total_tr_len]
total_tr_label = Y[:total_tr_len]

val_data = X[total_tr_len:(total_tr_len + val_len)]
val_label = Y[total_tr_len:(total_tr_len + val_len)]

te_data = X[(total_tr_len + val_len):(total_tr_len + val_len + te_len)]
te_label = Y[(total_tr_len + val_len):(total_tr_len + val_len + te_len)]

total_tr_data_tensor = torch.from_numpy(total_tr_data).type(torch.FloatTensor)
total_tr_label_tensor = torch.from_numpy(total_tr_label).type(torch.LongTensor)

val_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor)
val_label_tensor = torch.from_numpy(val_label).type(torch.LongTensor)

te_data_tensor = torch.from_numpy(te_data).type(torch.FloatTensor)
te_label_tensor = torch.from_numpy(te_label).type(torch.LongTensor)

print('total tr len %d | val len %d | test len %d' % (
len(total_tr_data_tensor), len(val_data_tensor), len(te_data_tensor)))

# ==============================================================================================================

user_tr_data_tensors = []
user_tr_label_tensors = []

for i in range(nusers):
    user_tr_data_tensor = torch.from_numpy(total_tr_data[user_tr_len * i:user_tr_len * (i + 1)]).type(torch.FloatTensor)
    user_tr_label_tensor = torch.from_numpy(total_tr_label[user_tr_len * i:user_tr_len * (i + 1)]).type(
        torch.LongTensor)

    user_tr_data_tensors.append(user_tr_data_tensor)
    user_tr_label_tensors.append(user_tr_label_tensor)
    print('user %d tr len %d' % (i, len(user_tr_data_tensor)))

batch_size = 250
resume = 0
schedule = [1000]
nbatches = user_tr_len // batch_size

gamma = .5
opt = 'sgd'
fed_lr = 0.5
criterion = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()

aggregation = 'median'
multi_k = False
candidates = []

dev_type = 'std'
z_values = {3: 0.69847, 5: 0.7054, 8: 0.71904, 10: 0.72575, 12: 0.73891}
n_attackers = [0]

arch = 'alexnet'
chkpt = './' + aggregation

results = []

server_control_dict = {0: [0, 1, 2, 3, 4, 5], 1: [1, 2, 5, 6, 7, 8], 2: [3, 4, 5, 7, 8, 9]}
overlap_weight_index = {0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 1, 7: 2, 8: 2, 9: 1}

for n_attacker in n_attackers:
    epoch_num = 0
    best_global_acc = 0
    best_global_te_acc = 0

    clients = []

    # Create clients
    print("creating %d clients" % (nusers))
    for i in range(nusers):
        if i < nusers - n_attacker:
            clients.append(Client(i, False, arch, fed_lr, criterion))
        else:
            clients.append(Client(i, True, arch, fed_lr, criterion))

    # torch.cuda.empty_cache()
    r = np.arange(user_tr_len)
    while epoch_num <= args.epochs:
        user_grads = []

        # Shuffle data for each epoch except the first one
        if not epoch_num and epoch_num % nbatches == 0:
            np.random.shuffle(r)
            for i in range(nusers):
                user_tr_data_tensors[i] = user_tr_data_tensors[i][r]
                user_tr_label_tensors[i] = user_tr_label_tensors[i][r]

        # Iterate over users, excluding attackers
        for i in range(n_attacker, nusers):
            # Get a batch of inputs and targets for the current user
            inputs = user_tr_data_tensors[i][
                     (epoch_num % nbatches) * batch_size:((epoch_num % nbatches) + 1) * batch_size]
            targets = user_tr_label_tensors[i][
                      (epoch_num % nbatches) * batch_size:((epoch_num % nbatches) + 1) * batch_size]

            param_grad = clients[i].train(inputs, targets)

            # Concatenate user gradients to the list
            user_grads = param_grad[None, :] if len(user_grads) == 0 else torch.cat((user_grads, param_grad[None, :]),
                                                                                    0)

        # Store the collected user gradients as malicious gradients
        malicious_grads = user_grads


        # # Check if the current epoch is in the specified schedule
        # if epoch_num in schedule:
        #     # Iterate over parameter groups in the optimizer
        #     for param_group in optimizer_fed.param_groups:
        #         # Update the learning rate of each parameter group using the specified decay factor (gamma)
        #         param_group['lr'] *= gamma
        #
        #         # Print the updated learning rate
        #         print('New learning rate:', param_group['lr'])

        # Add the parameters of the malicious clients depending on attack type
        if n_attacker > 0:
            if args.attack == 'lie':
                mal_update = lie_attack(malicious_grads, z_values[n_attacker])
                malicious_grads = torch.cat((torch.stack([mal_update] * n_attacker), malicious_grads))
            elif args.attack == 'fang':
                agg_grads = torch.mean(malicious_grads, 0)
                deviation = torch.sign(agg_grads)
                malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
            elif args.attack == 'agr':
                agg_grads = torch.mean(malicious_grads, 0)
                malicious_grads = min_max_attack(malicious_grads, agg_grads, n_attacker, dev_type=dev_type)

        if not epoch_num:
            print(malicious_grads.shape)

        # Store of aggregate gradients for servers
        server_aggregates = []

        # For each server find the aggregate gradients of clients it reaches
        for server in server_control_dict.keys():
            clients_in_reach = []
            for clientId in server_control_dict[server]:
                clients_in_reach.append(malicious_grads[clientId])

            agg_grads = []
            stacked_clients_in_reach = torch.stack(clients_in_reach, dim=0)
            if aggregation == 'median':
                agg_grads = fedmes_median(stacked_clients_in_reach, overlap_weight_index)

            elif aggregation == 'average':
                agg_grads = fedmes_mean(stacked_clients_in_reach, overlap_weight_index)

            server_aggregates.append(agg_grads)

        del user_grads

        del malicious_grads

        server_aggregates = torch.stack(server_aggregates, dim=0)

        # Update models of clients taking into account the servers that reach it
        for client in clients:
            if client.client_idx == 0:
                client.update_model(server_aggregates[0])
            elif client.client_idx == 6:
                client.update_model(server_aggregates[1])
            elif client.client_idx == 9:
                client.update_model(server_aggregates[2])
            elif client.client_idx == 5:
                comb_all = torch.mean(server_aggregates, dim=0)
                client.update_model(comb_all)
            elif client.client_idx in [1, 2]:
                comb_0_1 = torch.mean(server_aggregates[:2], dim=0)
                client.update_model(comb_0_1)
            elif client.client_idx in [7, 8]:
                comb_1_2 = torch.mean(server_aggregates[1:], dim=0)
                client.update_model(comb_1_2)
            elif client.client_idx in [3, 4]:
                comb_0_2 = torch.mean(server_aggregates[[0, 2]], dim=0)
                client.update_model(comb_0_2)

        val_loss, val_acc = test(val_data_tensor, val_label_tensor, clients[5].fed_model, criterion, use_cuda)
        te_loss, te_acc = test(te_data_tensor, te_label_tensor, clients[5].fed_model, criterion, use_cuda)

        is_best = best_global_acc < val_acc

        best_global_acc = max(best_global_acc, val_acc)

        if is_best:
            best_global_te_acc = te_acc

        print("Acc: " + str(val_acc) + " Loss: " + str(val_loss))
        results.append([val_acc, val_loss])
        if epoch_num % 10 == 0 or epoch_num == args.epochs - 1:
            print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f' % (
            aggregation, args.attack, n_attacker, epoch_num, val_loss, val_acc, best_global_acc, best_global_te_acc))

        if val_loss > 10:
            print('val loss %f too high' % val_loss)
            break

        epoch_num += 1

print(results)
print("Saving to results.csv")

with open("results.csv", 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the data rows
    csvwriter.writerows(results)

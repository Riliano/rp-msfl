from __future__ import print_function

import os
import pickle
import csv

import numpy as np

import torch.nn as nn

from utils.misc import get_time_string

from aggregation_fedmes import fedmes_median, fedmes_mean
from aggregation_single_server import *

from cifar10.cifar10_normal_train import *
from cifar10.cifar10_models import *
from cifar10.sgd import SGD

from arguments import Arguments

from data import TrainingTensors, tensor_loader

from attack import min_max_attack, lie_attack, get_malicious_updates_fang_trmean, our_attack_dist
from client import Client

args = Arguments()

tensors = tensor_loader(args)

nbatches = args.user_tr_len // args.batch_size

criterion = nn.CrossEntropyLoss()

multi_k = (args.aggregation == 'mkrum')
candidates = []

chkpt = './' + args.topology + '-' + args.aggregation

results_file = './results/' + get_time_string()       + '-'\
                            + args.topology           + '-'\
                            + args.aggregation        + '-'\
                            + str(args.epochs)        +'e-'\
                            + str(args.num_attackers) + 'att-'\
                            + args.arch               + '.csv'
results = []
print('Results will be saved in: ' + results_file)
with open(results_file, 'w') as csvfile:
    csvwriter = csv.writer(csvfile).writerow(['Accuracy', 'Loss'])

# Keep track of the clients each server reaches
server_control_dict = {0: [0, 1, 2, 3, 4, 5], 1: [1, 2, 0, 6, 7, 8], 2: [3, 4, 0, 7, 8, 9]}
# Keep track of weights
overlap_weight_index = {0: 3, 1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 2, 8: 2, 9: 1}

epoch_num = 0
best_global_acc = 0
best_global_te_acc = 0

clients = []

# Create clients
print("creating %d clients" % (args.clients))
for i in range(args.clients):
    if i >= args.num_attackers:
        clients.append(Client(i, False, args.arch, args.fed_lr, criterion))
    else:
        clients.append(Client(i, True, args.arch, args.fed_lr, criterion))

# torch.cuda.empty_cache()
r = np.arange(args.user_tr_len)
while epoch_num < args.epochs:
    user_grads = []

    # Shuffle data for each epoch except the first one
    if not epoch_num and epoch_num % nbatches == 0:
        np.random.shuffle(r)
        for i in range(args.clients):
            tensors.user_tr_data[i] = tensors.user_tr_data[i][r]
            tensors.user_tr_label[i] = tensors.user_tr_label[i][r]

    # Iterate over users, excluding attackers
    for i in range(args.num_attackers, args.clients):
        # Get a batch of inputs and targets for the current user
        inputs = tensors.user_tr_data[i][
                 (epoch_num % nbatches) * args.batch_size:((epoch_num % nbatches) + 1) * args.batch_size]
        targets = tensors.user_tr_label[i][
                  (epoch_num % nbatches) * args.batch_size:((epoch_num % nbatches) + 1) * args.batch_size]

        param_grad = clients[i].train(inputs, targets)

        # Concatenate user gradients to the list
        user_grads = param_grad[None, :] if len(user_grads) == 0 else torch.cat((user_grads, param_grad[None, :]),
                                                                                0)

    # Store the collected user gradients as malicious gradients
    malicious_grads = user_grads

    # Update learning rate of clients
    for client in clients:
        client.update_learning_rate(epoch_num, args.schedule, args.gamma)

    # Add the parameters of the malicious clients depending on attack type
    if args.num_attackers > 0:
        if args.attack == 'lie':
            mal_update = lie_attack(malicious_grads, args.z_values[args.num_attackers])
            malicious_grads = torch.cat((torch.stack([mal_update] * args.num_attackers), malicious_grads))
        elif args.attack == 'fang':
            agg_grads = torch.mean(malicious_grads, 0)
            deviation = torch.sign(agg_grads)
            malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, args.num_attackers, epoch_num)
        elif args.attack == 'agr':
            agg_grads = torch.mean(malicious_grads, 0)
            malicious_grads = our_attack_dist(malicious_grads, agg_grads, args.num_attackers, dev_type=args.dev_type)

    if not epoch_num:
        print(malicious_grads.shape)

    # Store of aggregate gradients for servers
    server_aggregates = []

    if args.topology == 'single':
        agg = []
        match args.aggregation:
            case 'median': agg = torch.median(malicious_grads,dim=0)[0] 
            case 'average': agg = torch.mean(malicious_grads,dim=0)
            case 'trmean': agg = tr_mean(malicious_grads, args.num_attackers)
            case 'krum': # TODO add support for mkrum or aggregation=='mkrum':
                #multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg, krum_candidate = multi_krum(malicious_grads, args.num_attackers, multi_k=multi_k)
            case 'bulyan': agg, krum_candidate=bulyan(malicious_grads, args.num_attackers)
            case _: assert (False), 'Unknown aggregation strategy: ' + self.aggregation
        server_aggregates.append(agg)
    elif args.topology == 'fedmes':
        # For each server find the aggregate gradients of clients it reaches
        for server in server_control_dict.keys():
            clients_in_reach = []
            for clientId in server_control_dict[server]:
                clients_in_reach.append(malicious_grads[clientId])

            agg_grads = []
            stacked_clients_in_reach = torch.stack(clients_in_reach, dim=0)
            if args.aggregation == 'median':
                agg_grads = fedmes_median(stacked_clients_in_reach, overlap_weight_index)

            elif args.aggregation == 'average':
                agg_grads = fedmes_mean(stacked_clients_in_reach, overlap_weight_index)

            server_aggregates.append(agg_grads)
    else:
        assert (False), 'Unkown topology: ' + args.topology

    del user_grads
    del malicious_grads

    server_aggregates = torch.stack(server_aggregates, dim=0)

    if args.topology == 'single':
        for c in clients:
            c.update_model(server_aggregates[0])
    elif args.topology == 'fedmes':
        # Update models of clients taking into account the servers that reach it
        for client in clients:
            if client.client_idx == 5:
                client.update_model(server_aggregates[0])
            elif client.client_idx == 6:
                client.update_model(server_aggregates[1])
            elif client.client_idx == 9:
                client.update_model(server_aggregates[2])
            elif client.client_idx == 0:
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

    val_loss, val_acc = test(tensors.val_data, tensors.val_label, clients[0].fed_model, criterion, args.cuda)
    te_loss, te_acc = test(tensors.te_data, tensors.te_label, clients[0].fed_model, criterion, args.cuda)

    is_best = best_global_acc < val_acc

    best_global_acc = max(best_global_acc, val_acc)

    if is_best:
        best_global_te_acc = te_acc

    print("Acc: " + str(val_acc) + " Loss: " + str(val_loss))
    results.append([val_acc, val_loss])
    if epoch_num % 10 == 0 or epoch_num == args.epochs - 1:
        print('%s, %s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f' % (
            args.topology, args.aggregation, args.attack, args.num_attackers, epoch_num, val_loss, val_acc, best_global_acc,
            best_global_te_acc))

    if args.batch_write and epoch_num % args.batch_write == 0:
        print('Writing next batch of results at e ' + str(epoch_num))
        with open(results_file, 'a') as csvfile:
            csv.writer(csvfile).writerows(results)
            results.clear()

    if val_loss > 10:
        print('val loss %f too high' % val_loss)
        break

    epoch_num += 1

print('Saving to ' + results_file)
with open(results_file, 'a') as csvfile:
    csv.writer(csvfile).writerows(results)

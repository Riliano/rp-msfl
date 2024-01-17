from __future__ import print_function

import os
import pickle
import csv

import numpy as np
import torch.nn as nn

from cifar10.cifar10_normal_train import *

from utils.misc import get_time_string
from arguments import Arguments
from data import tensor_loader

from aggregation_fedmes import fedmes_median, fedmes_mean
from aggregation_single_server import *
from attack import min_max_attack, lie_attack, get_malicious_updates_fang_trmean, minmax_ndss, veiled_minmax
from client import Client

def find_servers_in_reach(args, server_control_dict):
    client_reach_list = []
    overlap_weight_index = {}
    for i in range(args.clients):
        reach = []
        for k in server_control_dict:
            if i in server_control_dict[k]:
                reach.append(k)
        client_reach_list.append(reach)
        overlap_weight_index[i] = len(reach)
    return client_reach_list, overlap_weight_index


def run_experiment(args):
    loaders, validation = tensor_loader(args)
    criterion = nn.CrossEntropyLoss()

    multi_k = (args.aggregation == 'mkrum')
    candidates = []

    chkpt = './' + args.topology + '-' + args.aggregation

    results_file = './results/' + get_time_string()       + '-'\
                                + args.dataset            + '-'\
                                + args.topology           + '-'\
                                + args.aggregation        + '-'\
                                + str(args.epochs)        +'e-'\
                                + str(args.num_attackers) + 'att-'\
                                + args.attack             + '-'\
                                + args.arch               + '.csv'
    results = []
    if args.batch_write:
        print('Results will be saved in: ' + results_file)
        with open(results_file, 'w') as csvfile:
            csv.writer(csvfile).writerow(['Accuracy', 'Loss'])

    server_control_dict = args.server_control_dict
    client_server_reach, overlap_weight_index = find_servers_in_reach(args, server_control_dict)

    epoch_num = 0
    best_global_acc = 0
    best_global_te_acc = 0

    clients = []

    # Create clients
    print("creating %d clients" % (args.clients))
    if args.attacker_select == 'first-n':
        for i in range(0, args.num_attackers):
            clients.append(Client(i, args, loaders[i], True, criterion))
        for i in range(args.num_attackers, args.clients):
            clients.append(Client(i, args, loaders[i], False, criterion))
    elif args.attacker_select == 'id':
        for i in range(0, args.clients):
            clients.append(Client(i, args, loaders[i], (i in args.attacker_ids), criterion))
        args.num_attackers = len(args.attacker_ids)

    num_attackers = args.num_attackers
    args.num_attackers = 0
    if args.cuda:
        torch.cuda.empty_cache()

    while epoch_num < args.epochs:
        user_grads = []

        if epoch_num == args.epochs_before_attack:
            if num_attackers > 0:
                print('Activating malicious clients')
            args.num_attackers = num_attackers
        for c in clients:
            param_grad = c.train()
            user_grads = param_grad[None, :] if len(user_grads) == 0 else torch.cat((user_grads, param_grad[None, :]),
                                                                                    0)

        # Store the collected user gradients as malicious gradients
        malicious_grads = user_grads

        # Collect the gradients from all benign clients
        clean_grads = []
        for c in clients:
            if not c.is_mal:
                clean_grads.append(user_grads[c.client_idx])
        clean_grads = torch.stack(clean_grads, 0)

        # Update learning rate of clients
        for client in clients:
            client.update_learning_rate(epoch_num, args.schedule, args.gamma)

        # Add the parameters of the malicious clients depending on attack type
        if args.num_attackers > 0:
            if args.attack == 'lie':
                mal_update = lie_attack(clean_grads, args.z_values[args.num_attackers])
                malicious_grads = torch.cat((torch.stack([mal_update] * args.num_attackers), clean_grads))
            elif args.attack == 'fang':
                agg_grads = torch.mean(clean_grads, 0)
                deviation = torch.sign(agg_grads)
                malicious_grads = get_malicious_updates_fang_trmean(clean_grads, deviation, args.num_attackers, epoch_num)
            elif args.attack == 'minmax':
                agg_grads = torch.mean(clean_grads, 0)
                malicious_grads = minmax_ndss(clean_grads, agg_grads, args.num_attackers, dev_type=args.dev_type)
            elif args.attack == 'collab-minmax' and epoch_num:
                agg_grads = []
                mal_id = []
                for c in clients:
                    if c.is_mal:
                        mal_id.append(c.client_idx)
                        agg_grads.append(malicious_grads[c.client_idx])
                        for update in c.available_updates:
                            agg_grads.append(update)
                agg_grads = torch.stack(agg_grads, 0)
                mean = torch.mean(agg_grads, 0)
                m_grad = veiled_minmax(agg_grads, mean, dev_type=args.dev_type)
                if (epoch_num == 1):
                    for i in mal_id:
                        print('available updates ' + str(len(clients[i].available_updates)))
                        print('servers: ', client_server_reach[clients[i].client_idx])

                for i in mal_id:
                    malicious_grads[i] = m_grad

            elif args.attack == 'zerok-minmax' and epoch_num:
                for c in clients:
                    agg_grads = []
                    if c.is_mal:
                        agg_grads.append(malicious_grads[c.client_idx])
                        for update in c.available_updates:
                            agg_grads.append(update)

                        if (epoch_num == 1):
                            print('available updates ' + str(len(c.available_updates)))
                            print('servers: ', client_server_reach[c.client_idx])

                        agg_grads = torch.stack(agg_grads, 0)
                        m_grad = veiled_minmax(agg_grads, torch.mean(agg_grads, 0), dev_type=args.dev_type)
                        malicious_grads[c.client_idx] = m_grad
            elif args.attack == 'veiled-minmax' and epoch_num:
                # Store the malicious gradients in a dict, so that it's updated all at once
                # in the case of multiple attackers
                malicious_dict = {}
                for c in clients:
                    if c.is_mal:
                        ids_in_reach = []
                        # Figure out which clients are within the same cell as the attacker
                        # In the case of single server, that is all of them
                        if args.topology == 'fedmes':
                            for server_id in client_server_reach[c.client_idx]:
                                ids_in_reach = ids_in_reach + server_control_dict[server_id]
                        else:
                            ids_in_reach = list(range(0, args.clients))

                        # Remove duplicates
                        ids_in_reach = list(set(ids_in_reach))

                        if (epoch_num == 1):
                            print('reach of ' + str(c.client_idx))
                            print(ids_in_reach)

                        agg_grads = []
                        #for i in ids_in_reach:
                        for i in [c.client_idx]:
                            agg_grads.append(malicious_grads[i])
                            agg_grads.append(malicious_grads[i])
                        agg_grads = torch.stack(agg_grads, 0)
                        #m_grad = veiled_minmax(agg_grads, torch.mean(agg_grads, 0), dev_type=args.dev_type)
                        m_grad = veiled_minmax(agg_grads, c.previous_agg_grads, dev_type=args.dev_type)
                        malicious_dict[c.client_idx] = m_grad
                for k in malicious_dict:
                    malicious_grads[k] = malicious_dict[k]


        if not epoch_num:
            print(malicious_grads.shape)

        # Store of aggregate gradients for servers
        server_aggregates = []

        if args.topology == 'single':
            agg = []
            if args.aggregation == 'median':
                agg = torch.median(malicious_grads,dim=0)[0]
            elif args.aggregation == 'average':
                agg = torch.mean(malicious_grads,dim=0)
            elif args.aggregation == 'trmean':
                agg = tr_mean(malicious_grads, args.num_attackers)
            elif args.aggregation == 'krum' or args.aggregation == 'mkrum':
                multi_k = True if args.aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg, krum_candidate = multi_krum(malicious_grads, args.num_attackers, multi_k=multi_k)
            elif args.aggregation == 'bulyan':
                agg, krum_candidate=bulyan(malicious_grads, args.num_attackers)
            else:
                assert (False), 'Unknown aggregation strategy: ' + self.aggregation
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
            for c in clients:
                reach = client_server_reach[c.client_idx]
                comb = torch.mean(server_aggregates[reach], dim=0)
                c.available_updates = server_aggregates[reach]
                c.update_model(comb)
            #for client in clients:
            #    if client.client_idx == 5:
            #        client.update_model(server_aggregates[0])
            #    elif client.client_idx == 6:
            #        client.update_model(server_aggregates[1])
            #    elif client.client_idx == 9:
            #        client.update_model(server_aggregates[2])
            #    elif client.client_idx == 0:
            #        comb_all = torch.mean(server_aggregates, dim=0)
            #        client.update_model(comb_all)
            #    elif client.client_idx in [1, 2]:
            #        comb_0_1 = torch.mean(server_aggregates[:2], dim=0)
            #        client.update_model(comb_0_1)
            #    elif client.client_idx in [7, 8]:
            #        comb_1_2 = torch.mean(server_aggregates[1:], dim=0)
            #        client.update_model(comb_1_2)
            #    elif client.client_idx in [3, 4]:
            #        comb_0_2 = torch.mean(server_aggregates[[0, 2]], dim=0)
            #        client.update_model(comb_0_2)

        val_loss, val_acc = test(validation[0], validation[1], clients[0].fed_model, criterion, args.cuda)
        best_global_acc = max(best_global_acc, val_acc)

        print("Acc: " + str(val_acc) + " Loss: " + str(val_loss))
        results.append([val_acc, val_loss])
        if epoch_num % 10 == 0 or epoch_num == args.epochs - 1:
            print('%s, %s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f' % (
                args.topology, args.aggregation, args.attack, args.num_attackers, epoch_num, val_loss, val_acc, best_global_acc))

        if args.batch_write and epoch_num % args.batch_write == 0:
            print('Writing next batch of results at e ' + str(epoch_num))
            with open(results_file, 'a') as csvfile:
                csv.writer(csvfile).writerows(results)
                results.clear()

        if val_loss > 100:
            print('val loss %f too high' % val_loss)
            break

        epoch_num += 1


    print('Saving to ' + results_file)
    if args.batch_write:
        with open(results_file, 'a') as csvfile:
            csv.writer(csvfile).writerows(results)
    else:
        with open(results_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Accuracy', 'Loss'])
            csvwriter.writerows(results)
    if args.save_final_model:
        weights_file = './pretrained/' + get_time_string() + '-'\
                                       + args.arch         + '-'\
                                       + args.dataset      + '-'\
                                       + str(args.epochs)  + '.zip'
        print('Saving weights to ' + weights_file)
        torch.save(clients[0].fed_model.state_dict(), weights_file)
    print('Done')

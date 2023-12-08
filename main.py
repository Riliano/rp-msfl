from __future__ import print_function

import os
import pickle

import numpy as np

import torch.nn as nn


from cifar10.cifar10_normal_train import *

from cifar10.cifar10_models import *

from cifar10.sgd import SGD


from arguments import Arguments

from attack import min_max_attack, lie_attack, get_malicious_updates_fang_trmean

args = Arguments()

#X, Y = load_data(args)
data = pickle.load(open('./cifar10_data_ind.pkl', 'rb'))
X = data[0]
Y = data[1]
# data loading

nusers=args.clients
user_tr_len=2400

total_tr_len=user_tr_len*nusers
val_len=3300
te_len=3300

print('total data len: ',len(X))

if not os.path.isfile('./cifar10_shuffle.pkl'):
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open('./cifar10_shuffle.pkl','wb'))
else:
    all_indices=pickle.load(open('./cifar10_shuffle.pkl','rb'))

total_tr_data=X[:total_tr_len]
total_tr_label=Y[:total_tr_len]

val_data=X[total_tr_len:(total_tr_len+val_len)]
val_label=Y[total_tr_len:(total_tr_len+val_len)]

te_data=X[(total_tr_len+val_len):(total_tr_len+val_len+te_len)]
te_label=Y[(total_tr_len+val_len):(total_tr_len+val_len+te_len)]

total_tr_data_tensor=torch.from_numpy(total_tr_data).type(torch.FloatTensor)
total_tr_label_tensor=torch.from_numpy(total_tr_label).type(torch.LongTensor)

val_data_tensor=torch.from_numpy(val_data).type(torch.FloatTensor)
val_label_tensor=torch.from_numpy(val_label).type(torch.LongTensor)

te_data_tensor=torch.from_numpy(te_data).type(torch.FloatTensor)
te_label_tensor=torch.from_numpy(te_label).type(torch.LongTensor)

print('total tr len %d | val len %d | test len %d'%(len(total_tr_data_tensor),len(val_data_tensor),len(te_data_tensor)))

#==============================================================================================================

user_tr_data_tensors=[]
user_tr_label_tensors=[]

for i in range(nusers):

    user_tr_data_tensor=torch.from_numpy(total_tr_data[user_tr_len*i:user_tr_len*(i+1)]).type(torch.FloatTensor)
    user_tr_label_tensor=torch.from_numpy(total_tr_label[user_tr_len*i:user_tr_len*(i+1)]).type(torch.LongTensor)

    user_tr_data_tensors.append(user_tr_data_tensor)
    user_tr_label_tensors.append(user_tr_label_tensor)
    print('user %d tr len %d'%(i,len(user_tr_data_tensor)))




batch_size=250
resume=0
schedule=[1000]
nbatches = user_tr_len//batch_size

gamma=.5
opt = 'sgd'
fed_lr=0.5
criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()

aggregation='median'
multi_k = False
candidates = []

dev_type = 'std'
z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
n_attackers=[0]

arch='alexnet'
chkpt='./'+aggregation

results = []

server_control_dict = {0: [0, 1, 2, 3, 4, 5], 1: [1, 2, 5, 6, 7, 8], 2: [3, 4, 5, 7, 8, 9]}
overlap_weight_index = {0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 1, 7: 2, 8: 2, 9: 1}

for n_attacker in n_attackers:
    epoch_num = 0
    best_global_acc = 0
    best_global_te_acc = 0

   # torch.cuda.empty_cache()
    r=np.arange(user_tr_len)

    fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
    optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

    while epoch_num <= args.epochs:
        user_grads=[]

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

            # Convert inputs and targets to PyTorch Variables
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # Forward pass
            outputs = fed_model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Zero out gradients, perform backward pass, and collect gradients
            fed_model.zero_grad()
            loss.backward(retain_graph=True)
            param_grad = []
            for param in fed_model.parameters():
                param_grad = param.grad.data.view(-1) if not len(param_grad) else torch.cat(
                    (param_grad, param.grad.view(-1)))

            # Concatenate user gradients to the list
            user_grads = param_grad[None, :] if len(user_grads) == 0 else torch.cat((user_grads, param_grad[None, :]),
                                                                                    0)

        # Store the collected user gradients as malicious gradients
        malicious_grads = user_grads

        # Check if the current epoch is in the specified schedule
        if epoch_num in schedule:
            # Iterate over parameter groups in the optimizer
            for param_group in optimizer_fed.param_groups:
                # Update the learning rate of each parameter group using the specified decay factor (gamma)
                param_group['lr'] *= gamma

                # Print the updated learning rate
                print('New learning rate:', param_group['lr'])

        # Add the parameters of the malicious clients depending on attack type
        if n_attacker > 0:
            if args.attack == 'lie':
                mal_update = lie_attack(malicious_grads, z_values[n_attacker])
                malicious_grads = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
            elif args.attack == 'fang':
                agg_grads = torch.mean(malicious_grads, 0)
                deviation = torch.sign(agg_grads)
                malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
            elif args.attack == 'agr':
                agg_grads = torch.mean(malicious_grads, 0)
                malicious_grads = min_max_attack(malicious_grads, agg_grads, n_attacker, dev_type=dev_type)

        if not epoch_num :
            print(malicious_grads.shape)

        # Aggregate gradients
        if aggregation=='median':
            agg_grads=torch.median(malicious_grads,dim=0)[0]

        elif aggregation=='average':
            agg_grads=torch.mean(malicious_grads,dim=0)

        # elif aggregation=='trmean':
        #     agg_grads= tr_mean(malicious_grads, n_attacker)
        #
        # elif aggregation=='krum' or aggregation=='mkrum':
        #     multi_k = True if aggregation == 'mkrum' else False
        #     if epoch_num == 0: print('multi krum is ', multi_k)
        #     agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)
        #
        # elif aggregation=='bulyan':
        #     agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

        del user_grads

        start_idx=0

        optimizer_fed.zero_grad()

        model_grads=[]

        for i, param in enumerate(fed_model.parameters()):
            param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
            start_idx=start_idx+len(param.data.view(-1))
            param_=param_#.cuda()
            model_grads.append(param_)

        optimizer_fed.step(model_grads)

        val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
        te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

        is_best = best_global_acc < val_acc

        best_global_acc = max(best_global_acc, val_acc)

        if is_best:
            best_global_te_acc = te_acc

        print("Acc: " + str(val_acc) + " Loss: " + str(val_loss))
        results.append([val_acc, val_loss])
        if epoch_num%10==0 or epoch_num==args.epochs-1:
            print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, args.attack, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

        if val_loss > 10:
            print('val loss %f too high'%val_loss)
            break

        epoch_num+=1

print(results)
print("Saving to results.csv")

import csv
with open("results.csv", 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the data rows
    csvwriter.writerows(results)

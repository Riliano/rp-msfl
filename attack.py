import torch


def get_malicious_updates_fang_trmean(all_updates, deviation, n_attackers, epoch_num, compression='none', q_level=2,
                                      norm='inf'):
    b = 2
    max_vector = torch.max(all_updates, 0)[0]
    min_vector = torch.min(all_updates, 0)[0]

    max_ = (max_vector > 0).type(torch.FloatTensor).cuda()
    min_ = (min_vector < 0).type(torch.FloatTensor).cuda()

    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
    min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
        [max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
        [min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

    quant_mal_vec = []
    if compression != 'none':
        if epoch_num == 0: print('compressing malicious update')
        for i in range(mal_vec.shape[0]):
            mal_ = mal_vec[i]
            mal_quant = qsgd(mal_, s=q_level, norm=norm)
            quant_mal_vec = mal_quant[None, :] if not len(quant_mal_vec) else torch.cat(
                (quant_mal_vec, mal_quant[None, :]), 0)
    else:
        quant_mal_vec = mal_vec

    mal_updates = torch.cat((quant_mal_vec, all_updates), 0)

    return mal_updates


def lie_attack(all_updates, z):
    avg = torch.mean(all_updates, dim=0)
    std = torch.std(all_updates, dim=0)
    return avg + z * std


def min_max_attack(all_updates, model_re, n_attackers, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0])  # .cuda() #compute_lambda_our(all_updates, model_re, n_attackers)

    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    iters = 0
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads = torch.median(mal_updates, 0)[0]

        loss = torch.norm(agg_grads - model_re)

        if prev_loss < loss:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss

    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)

    return mal_updates


'''
MIN-MAX attack
'''


def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec', threshold=30):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([threshold]).float()  # .cuda()
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)

    return mal_updates

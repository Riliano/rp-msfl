import torch

SEED = 1
torch.manual_seed(SEED)

class Arguments:
    def __init__(self):
        
        self.dataset = 'cifar_10'
        self.arch = 'alexnet'
        self.batch_size = 250
        self.schedule = [1000]
        self.gamma = 0.5
        self.fed_lr = 0.5
        self.dev_type = 'std'
        self.z_values = {3: 0.69847, 5: 0.7054, 8: 0.71904, 10: 0.72575, 12: 0.73891}


        self.resume = 0
        self.epochs = 800


        # How many epochs before the results are saved, disable with 0
        self.batch_write = 50

        self.clients = 10
        self.num_attackers = 0

        self.topology = "fedmes" # "single", "fedmes"
        self.aggregation = "average" # "average", "median"

        self.attack = "agr"

        self.cuda = False


        if self.dataset == "cifar_10":
            self.user_tr_len = 2400
            self.total_tr_len = self.user_tr_len * self.clients
            self.val_len = 3300
            self.te_len = 3300
#            #self.net = Cifar10CNN
#            # self.net = Cifar10ResNet
#
#            self.lr = 0.01
#            self.momentum = 0.5
#            self.scheduler_step_size = 50
#            self.scheduler_gamma = 0.5
#            self.min_lr = 1e-10
#            self.N = 50000
#            self.generator_image_num = 50
#            self.generator_local_epoch = 10
#            self.layer_image_num = 50
#            self.layer_image_epoch = 10
#            self.reduce = 1

#            self.train_data_loader_pickle_path = "data_loaders/cifar10/train_data_loader.pickle"
#            self.test_data_loader_pickle_path = "data_loaders/cifar10/test_data_loader.pickle"


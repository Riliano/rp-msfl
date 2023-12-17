import torch

SEED = 1
torch.manual_seed(SEED)

class Arguments:
    def __init__(self):
        
        self.dataset = "cifar_10"
        self.batch_size = 250
        self.resume = 0
        self.epochs = 10#800

        self.clients = 10

        self.attack = "agr"

        self.cuda = False

#
#        if self.dataset == "cifar_10":
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


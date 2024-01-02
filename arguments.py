import torch

SEED = 1
torch.manual_seed(SEED)

class Arguments:
    def __init__(self):
        
        self.dataset = 'cifar_10'
        self.arch = 'alexnet' # 'alexnet', 'vgg11', 'resnet-pretrained'

        self.load_pretrained_weights = True
        self.pretrained_weights_file = './pretrained/model-alexnet.zip'
        self.save_final_model = False

        self.batch_size = 250
        self.schedule = [1000]
        self.gamma = 0.5
        self.fed_lr = 0.2
        if self.arch == 'resnet-pretrained' or self.load_pretrained_weights:
            self.fed_lr = 0.00012

        self.dev_type = 'std'
        # Those values are taken from code with 50 clients, likely they need to be recomputed
        # See LIE attack paper(A Little Is Enough, Baruch et al) for more details
        # 1 and 2 are based on guessing and skimming the formulas
        self.z_values = {1: 0.7054, 2: 0.72575, 3: 0.69847, 5: 0.7054, 8: 0.71904, 10: 0.72575, 12: 0.73891}


        self.resume = 0
        self.epochs = 50
        self.epochs_before_attack = 0

        # How many epochs before the results are saved, disable with 0
        self.batch_write = 0

        self.clients = 10

        self.num_attackers = 2
        self.attacker_ids = [3, 4]
        # If attacker_select is set to 'id', num_attackers is ignored and adjusted
        # to the size of attacker_id, if 'first-n' is selected, 'attacer_ids' does nothing
        # This only matters for 'vailed-minmax' attack, the rest assume 'first-n' and will
        # probably break
        self.attacker_select = 'id' # 'first-n', 'id'

        self.topology = 'single' # 'single', 'fedmes'
        self.aggregation = 'average' # 'average', 'median'

        self.attack = 'minmax' # 'minmax', 'fang', 'lie', 'veiled-minmax'

        self.cuda = False
        self.parallel = True


        if self.dataset == "cifar_10":
            self.user_tr_len = 4000
            self.total_tr_len = self.user_tr_len * self.clients
            self.val_len = 10000
            self.te_len = 10000
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


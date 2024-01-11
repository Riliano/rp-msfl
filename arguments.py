import torch

SEED = 1
torch.manual_seed(SEED)

class Arguments:
    def __init__(self):
        
        self.available_datasets = ['cifar10', 'fashionmnist']
        self.dataset = self.available_datasets[1]
        self.arch = 'alexnet' # 'alexnet', 'vgg11', 'resnet-pretrained'

        self.load_pretrained_weights = False
        self.pretrained_weights_file = './pretrained/model-alexnet.zip'
        self.save_final_model = False

        self.batch_size = 256
        self.schedule = [60, 120, 240, 340, 420, 1000]

        self.gamma = 0.7
        self.fed_lr = 0.4
        if self.arch == 'resnet-pretrained' or self.load_pretrained_weights:
            self.fed_lr = 0.00082

        self.dev_type = 'std'
        # Those values are taken from code with 50 clients, likely they need to be recomputed
        # See LIE attack paper(A Little Is Enough..., Baruch et al) for more details
        # 1 and 2 are based on guessing and skimming the formulas
        self.z_values = {1: 0.7054, 2: 0.72575, 3: 0.69847, 5: 0.7054, 8: 0.71904, 10: 0.72575, 12: 0.73891}


        self.resume = 0
        self.epochs = 1200
        self.epochs_before_attack = 0

        # How many epochs before the results are saved, disable with 0
        self.batch_write = 0

        self.clients = 10

        self.num_attackers = 0
        self.attacker_ids = [3, 4]
        # If attacker_select is set to 'id', num_attackers is ignored and adjusted
        # to the size of attacker_id, if 'first-n' is selected, 'attacer_ids' does nothing
        # This only matters for 'vailed-minmax' attack, the rest assume 'first-n' and will
        # probably break
        self.attacker_select = 'first-n' # 'first-n', 'id'

        self.topology = 'single' # 'single', 'fedmes'
        self.aggregation = 'average' # 'average', 'median'

        self.attack = 'minmax' # 'minmax', 'fang', 'lie', 'veiled-minmax'

        self.cuda = False
        self.parallel = True

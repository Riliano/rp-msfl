import torch

SEED = 1
torch.manual_seed(SEED)

class Arguments:
    def __init__(self):
        
        self.available_datasets = ['fashionmnist']
        self.dataset = self.available_datasets[0]
        self.arch = 'fashioncnn' # 'alexnet', 'vgg11', 'resnet-pretrained', 'fashioncnn'

        self.load_pretrained_weights = False
        self.pretrained_weights_file = './pretrained/model-alexnet.zip'
        self.save_final_model = False

        self.batch_size = 256
        self.schedule = [60, 120, 240, 340, 420, 1000]

        self.gamma = 0.7
        self.fed_lr = 0.006
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

        self.num_attackers = 2
        self.attacker_ids = [3, 7]
        # If attacker_select is set to 'id', num_attackers is ignored and adjusted
        # to the size of attacker_id, if 'first-n' is selected, 'attacer_ids' does nothing
        # This only matters for 'vailed-minmax' attack, the rest assume 'first-n' and will
        # probably break
        self.attacker_select = 'id' # 'first-n', 'id'

        self.topology = 'fedmes' # 'single', 'fedmes'
        self.topology_variant = 'dense' # 'sparse'
        if (self.topology_variant == 'dense'):
            self.server_control_dict = {0: [0, 1, 2, 3, 4, 5], 1: [1, 2, 0, 6, 7, 8], 2: [3, 4, 0, 7, 8, 9]}
        elif (self.topology_variant == 'sparse'):
            self.server_control_dict = {0: [0, 1, 2, 3, 4, 5], 1: [3, 4, 6, 7, 8], 2: [0, 1, 9]}

        self.aggregation = 'average' # 'average', 'median'

        self.attack = 'collab-minmax' # 'minmax', 'fang', 'lie', 'veiled-minmax' 'zerok-minmax'

        self.cuda = False
        self.parallel = True

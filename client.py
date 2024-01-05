import torch
from cifar10.cifar10_models import return_model

class Client:
    def __init__(self, client_idx, args, data_loader, is_mal, criterion):
        self.client_idx = client_idx
        self.args = args
        self.data_loader = data_loader
        self.is_mal = is_mal
        self.model_type = self.args.arch
        self.fed_lr = self.args.fed_lr
        self.criterion = criterion

        self.fed_model, self.optimizer_fed = return_model(self.model_type,\
                                                          lr=args.fed_lr,\
                                                          momentum=0.9,\
                                                          parallel=args.parallel,\
                                                          cuda=args.cuda)
        self.data_loader_iter = iter(self.data_loader)
        self.data_loader_i = 0
        if (args.load_pretrained_weights):
            print('Loading pretrained weights from: ' + args.pretrained_weights_file)
            self.fed_model.load_state_dict(torch.load(args.pretrained_weights_file), strict=False)

    def train(self):#, inputs, targets):
        if (self.data_loader_i == len(self.data_loader)):
            self.data_loader_i = 0
            self.data_loader_iter = iter(self.data_loader)

        # Convert inputs and labels to PyTorch Variables
        inputs, targets = next(self.data_loader_iter)#torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        self.data_loader_i = self.data_loader_i + 1

        # Forward pass
        outputs = self.fed_model(inputs)

        # Compute loss
        loss = self.criterion(outputs, targets)

        # Zero out gradients, perform backward pass, and collect gradients
        self.fed_model.zero_grad()
        loss.backward(retain_graph=True)
        param_grad = []
        for param in self.fed_model.parameters():
            param_grad = param.grad.data.view(-1) if not len(param_grad) else torch.cat(
                (param_grad, param.grad.view(-1)))

        return param_grad

    def update_model(self, agg_grads):
        self.previous_agg_grads = agg_grads
        # Initialize the starting index for aggregating gradients
        start_idx = 0

        # Zero out the gradients in the optimizer to avoid accumulation
        self.optimizer_fed.zero_grad()

        # List to store model gradients
        model_grads = []

        # Iterate over model parameters and get new gradients for parameters which exist in this model
        for i, param in enumerate(self.fed_model.parameters()):
            # Extract a slice of aggregated gradients corresponding to the current parameter
            param_ = agg_grads[start_idx:start_idx + len(param.data.view(-1))].reshape(param.data.shape)
            start_idx = start_idx + len(param.data.view(-1))
            param_ = param_  # .cuda()
            model_grads.append(param_)

        # Perform a step in the optimizer using the aggregated gradients
        self.optimizer_fed.step(model_grads)

    def update_learning_rate(self, epoch_num, schedule, gamma):
        if epoch_num in schedule:
            # Iterate over parameter groups in the optimizer
            for param_group in self.optimizer_fed.param_groups:
                # Update the learning rate of each parameter group using the specified decay factor (gamma)
                param_group['lr'] *= gamma

                # Print the updated learning rate
                print('New learning rate:', param_group['lr'])

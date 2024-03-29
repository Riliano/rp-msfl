from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel
import models.cifar as models
from cifar10.sgd import SGD
from cifar10.adam import Adam

class cifar_mlp(nn.Module):
    def __init__(self, ninputs=3 * 32 * 32, num_classes=10):
        super(cifar_mlp, self).__init__()
        self.ninputs = ninputs
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Linear(self.ninputs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, self.ninputs)
        hidden_out = self.features(x)
        return self.classifier(hidden_out)

def get_model(config, parallel=True, cuda=False, device=0):
    # print("==> creating model '{}'".format(config['arch']))
    if config['arch'].startswith('resnext'):
        model = models.__dict__[config['arch']](
            cardinality=config['cardinality'],
            num_classes=config['num_classes'],
            depth=config['depth'],
            widen_factor=config['widen-factor'],
            dropRate=config['drop'],
        )
    elif config['arch'].startswith('densenet'):
        model = models.__dict__[config['arch']](
            num_classes=config['num_classes'],
            depth=config['depth'],
            growthRate=config['growthRate'],
            compressionRate=config['compressionRate'],
            dropRate=config['drop'],
        )
    elif config['arch'].startswith('wrn'):
        model = models.__dict__[config['arch']](
            num_classes=config['num_classes'],
            depth=config['depth'],
            widen_factor=config['widen-factor'],
            dropRate=config['drop'],
        )
    elif config['arch'].endswith('resnet'):
        model = models.__dict__[config['arch']](
            num_classes=config['num_classes'],
            depth=config['depth'],
        )
    elif config['arch'].endswith('convnet'):
        model = models.__dict__[config['arch']](
            num_classes=config['num_classes']
        )
    else:
        model = models.__dict__[config['arch']](num_classes=config['num_classes'], )

    if parallel:
        model = torch.nn.DataParallel(model)

    if cuda:
        model.cuda()

    return model


def return_model(model_name, lr, momentum, parallel=False, cuda=True, device=0):
    if model_name == 'resnet-pretrained':
        # Note: This might get depricated in the near future, if this is the case
        # you might be able to still download the models from:
        # https://github.com/chenyaofo/pytorch-cifar-models
        # and manually import them
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet32', pretrained=True)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif model_name == 'dc':
        arch_config = {
            'arch': 'Dc',
            'num_classes': 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    elif model_name == 'alexnet':
        arch_config = {
            'arch': 'alexnet',
            'num_classes': 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    elif model_name == 'densenet-bc-100-12':
        arch_config = {
            'arch': 'densenet',
            'depth': 100,
            'growthRate': 12,
            'compressionRate': 2,
            'drop': 0,
            'num_classes': 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        # optimizer = SGD(model.parameters(), lr=0.1, momentum=momentum,weight_decay=1e-4)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif model_name == 'densenet-bc-L190-k40':
        arch_config = {
            'arch': 'densenet',
            'depth': 190,
            'growthRate': 40,
            'compressionRate': 2,
            'drop': 0,
            'num_classes': 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
    elif model_name == 'preresnet-110':
        arch_config = {
            'arch': 'preresnet',
            'depth': 110,
            'num_classes': 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        # optimizer = SGD(model.parameters(), lr=0.1, momentum=momentum, weight_decay=1e-4)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif model_name == 'resnet-110':
        arch_config = {
            'arch': 'resnet',
            'depth': 110,
            'num_classes': 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        # optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif model_name == 'resnext-16x64d':
        arch_config = {
            'arch': 'resnext',
            'depth': 29,
            'cardinality': 16,
            'widen-factor': 4,
            'drop': 0,
            'num_classes': 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    elif model_name == 'resnext-8x64d':
        arch_config = {
            'arch': 'resnext',
            'depth': 29,
            'cardinality': 8,
            'widen-factor': 4,
            'drop': 0,
            'num_classes': 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    elif model_name.startswith('vgg'):
        arch_config = {
            'arch': model_name,
            'num_classes': 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    elif model_name == 'WRN-28-10-drop':
        arch_config = {
            'arch': 'wrn',
            'depth': 28,
            'widen-factor': 10,
            'drop': 0.3,
            'num_classes': 10,
        }
        model = get_model(arch_config, parallel=parallel, cuda=cuda, device=device)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    else:
        assert (False), ('Model not found: ' + model_name)

    return model, optimizer

import numpy as np


class SGD:

    def __init__(self, sequential, lr, momentum=0.9, weight_decay=0, lr_decay=1):
        self.sequential = sequential
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v_dict = {}
        self.count = 0
        self.lr_decay = lr_decay
        for name, layer in self.sequential.layers.items():
            try:
                params = layer.parameters
                self.v_dict[name] = {}
                for param in params.keys():
                    self.v_dict[name][param] = 0
            except:
                continue

    def step(self):
        lr = self.lr * (self.lr_decay ** self.count)
        self.count += 1
        for layer_name, layer in self.sequential.layers.items():
            try:
                for param in layer.parameters.keys():
                    self.v_dict[layer_name][param] = self.momentum * self.v_dict[layer_name][param] + (
                                1 - self.momentum) * layer.grads[param]
                    v_corrected = self.v_dict[layer_name][param] / (1 - self.momentum ** self.count)
                    layer.parameters[param] = (1 - self.weight_decay) * layer.parameters[param] - self.lr * v_corrected
            except:
                continue


class Adam:

    def __init__(self, sequential, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.count = 0
        self.sequential = sequential
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.v_dict = {}
        self.s_dict = {}
        for name, layer in self.sequential.layers.items():
            try:
                params = layer.parameters
                self.v_dict[name] = {}
                self.s_dict[name] = {}
                for param in params.keys():
                    self.v_dict[name][param] = 0
                    self.s_dict[name][param] = 0
            except:
                continue

    def step(self):
        self.count += 1
        for layer_name, layer in self.sequential.layers.items():
            try:
                for param in layer.parameters.keys():
                    self.v_dict[layer_name][param] = self.beta1 * self.v_dict[layer_name][param] + (
                                1 - self.beta1) * layer.grads[param]
                    v_corrected = self.v_dict[layer_name][param] / (1 - self.beta1 ** self.count)
                    self.s_dict[layer_name][param] = self.beta2 * self.s_dict[layer_name][param] + (
                                1 - self.beta2) * layer.grads[param] ** 2
                    s_corrected = self.s_dict[layer_name][param] / (1 - self.beta2 ** self.count)
                    layer.parameters[param] = (1 - self.weight_decay) * layer.parameters[param] - \
                                              self.lr * v_corrected / (np.sqrt(s_corrected) + self.eps)
            except:
                continue
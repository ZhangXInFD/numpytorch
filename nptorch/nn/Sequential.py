

class Modules:

    def __init__(self):
        raise NotImplementedError

    def forward(self, x, eval_pattern):
        for layer in self.layers.values():
            if eval_pattern and layer.__class__.__name__ == 'Droupout':
                continue
            x = layer(x)
        return x

    # def eval(self):
    #     self.eval_method = True
    #
    # def train(self):
    #     self.eval_method = False

    def backward(self, dl):
        for name, layer in reversed(self.layers.items()):
            if name == 'conv2d1':
                dl = layer.backward(dl, backward=False)
            else:
                dl = layer.backward(dl)

    def __call__(self, x, eval_pattern=False):
        return self.forward(x, eval_pattern=eval_pattern)

    def state_dict(self):
        return self.parameters

    def save_state_dict(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.parameters, f, pickle.HIGHEST_PROTOCOL)
        return

    def load_state_dict(self, state_dict=None, path=None):
        if not state_dict:
            import pickle
            with open(path, 'rb') as f:
                state_dict = pickle.load(f)
        for layer in state_dict.keys():
            for param, v in state_dict[layer].items():
                self.parameters[layer][param] = v
        return


class Sequential(Modules):

    def __init__(self, architecture):
        from collections import OrderedDict
        if not isinstance(architecture, OrderedDict):
            cdict = {}
            self.layers = OrderedDict()
            self.parameters = OrderedDict()
            for layer in architecture:
                name = layer.__class__.__name__
                try:
                    count = cdict[name] + 1
                except:
                    count = 1
                cdict[name] = count
                self.layers[name.lower() + str(count)] = layer
                try:
                    self.parameters[name.lower() + str(count)] = layer.parameters
                except:
                    pass
                # self.name = []
                # for l, c in cdict.items():
                #     self.name.append(str(c))
                #     self.name.append(l)
                # self.name = ''.join(self.name)
        else:
            self.layers = architecture




import numpy as np
from .Loss import Loss


class Layer:

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dz):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class Linear(Layer):

    def __init__(self, inputs_dim, outputs_dim, bias=True):
        self.parameters = {}
        self.grads = {}
        self.parameters['w'] = np.random.randn(inputs_dim, outputs_dim) * 0.01
        if bias:
            self.parameters['b'] = np.zeros((1, outputs_dim))
        self.bias = bias

    def forward(self, x):
        self.cache = [x]
        if self.bias:
            return np.matmul(x, self.parameters['w']) + self.parameters['b']
        else:
            return np.matmul(x, self.parameters['w'])

    def backward(self, dz):
        x = self.cache[0]
        self.grads['w'] = np.matmul(x.T, dz) / x.shape[0]
        if self.bias:
            self.grads['b'] = np.sum(dz, axis=0) / x.shape[0]
        dx = np.matmul(dz, self.parameters['w'].T)
        return dx


class Conv2d(Layer):

    def __init__(self, kernel_size, in_channels, out_channels, padding=0, stride=1, bias=True):
        self.parameters = {}
        #glorot = np.sqrt(2 / ((in_channels + out_channels) * kernel_size[0] * kernel_size[1]))
        self.parameters['kernel'] = np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * \
                                    np.sqrt(2 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.bias = bias
        if self.bias:
            self.parameters['b'] = np.zeros((1, 1, 1, out_channels))
        self.padding = padding
        self.stride = stride
        self.grads = {}

    def forward(self, inputs):
        x = inputs.copy()
        outputs, stack = self.__conv(x=x, kernel=self.parameters['kernel'], padding=self.padding, stride=self.stride)
        self.cache = [stack]
        if self.bias:
            outputs += self.parameters['b']
        return outputs

    @classmethod
    def __conv(cls, x, kernel, padding, stride):  # x: m * height * width * channel
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), constant_values=0)
        m, xh, xw = x.shape[:3]
        x = np.transpose(x, (0, 3, 1, 2))
        out_channels, in_channels, kh, kw = kernel.shape
        line_kernel = kernel.reshape((out_channels, in_channels, kh * kw, 1))
        line_kernel = np.expand_dims(line_kernel, 1).repeat(m, axis=1)
        out_height = (xh - kh) // stride + 1
        out_width = (xw - kw) // stride + 1
        stack = cls.img2col(x, kw, kh, stride)
        outputs = np.transpose(np.sum(stack @ line_kernel, axis=2).reshape((out_channels, m, out_height, out_width)),
                               (1, 2, 3, 0))
        return outputs, stack  # outputs: m * height * width * channels

    @staticmethod
    def img2col(img, kw, kh, stride):
        m, in_channels, xh, xw = img.shape
        flag = True
        for h in range(0, xh + 1 - kh, stride):
            for w in range(0, xw + 1 - kw, stride):
                tmp = img[:, :, h:(h + kh), w:(w + kw)].reshape(m, in_channels, 1, kh * kw)
                if flag:
                    stack = tmp
                    flag = False
                else:
                    stack = np.concatenate((stack, tmp), axis=2)
        return stack

    @staticmethod
    def stride_padding(inputs, stride):  # inputs: m, channels, height, width
        m, channels, height, width = inputs.shape
        padn = stride - 1
        paded = np.append(inputs.reshape((m, channels, height, width, 1)), np.zeros((m, channels, height, width, padn)),
                          axis=4).reshape((m, channels, height, width * (padn + 1)))[:, :, :, :-padn]
        m, channels, height, width = paded.shape
        paded = np.append(paded.transpose(0, 1, 3, 2).reshape((m, channels, width, height, 1)),
                          np.zeros((m, channels, width, height, padn)), axis=4).reshape(
            (m, channels, width, height * (padn + 1)))[:, :, :, :-padn].transpose(0, 3, 2, 1)
        return paded  # inputs: m, height, width, channels

    def backward(self, dz, backward=True):  # dz m * height * width * out_channels
        if self.bias:
            self.grads['b'] = np.mean(dz, axis=(0, 1, 2))
        m, height, width, out_channels = dz.shape
        in_channels, kh, kw = self.parameters['kernel'].shape[1:]
        ds = np.transpose(dz, (3, 0, 1, 2))
        ds = ds.reshape((out_channels, m, height * width, 1))
        ds = np.expand_dims(ds, 2).repeat(in_channels, axis=2) # out_channels * m * in_channels * (height * width) * 1
        stack = self.cache[0]  # m * in_channels * (height * width)  * (kh * kw)
        dw = (np.transpose(stack, (0, 1, 3, 2)) @ ds).reshape((out_channels, m, in_channels, kh, kw))
        self.grads['kernel'] = np.mean(dw, axis=1)
        if backward:
            if self.stride != 1:
                dz = np.transpose(dz, (0, 3, 1, 2))  # m * out_channels * height * width
                dz = self.stride_padding(dz, self.stride)
            kernel = self.parameters['kernel'][:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
            padding = kernel.shape[2] - self.padding - 1
            dz, _ = self.__conv(x=dz, kernel=kernel, padding=padding, stride=1)
            return dz


class MaxPool2d(Layer):

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs):
        kh, kw = self.kernel_size
        m, xh, xw, channels = inputs.shape
        x = np.transpose(inputs, (0, 3, 1, 2))
        stack = Conv2d.img2col(img=x, kw=kw, kh=kh, stride=self.stride)
        out_height, out_width = (xh - kh) // self.stride + 1, (xw - kw) // self.stride + 1
        arg = np.eye(kh * kw)[np.argmax(stack, axis=3)]
        self.cache = [arg]
        outputs = np.max(stack, axis=3).reshape((m, channels, out_height, out_width)).transpose(0, 2, 3, 1)
        return outputs

    @staticmethod
    def col2img(col, kh, kw, h, w):
        m, c, steps = col.shape[:3]
        outputs = np.zeros((m, c, h, w))
        wc = w // kw
        for i in range(steps):
            hi, wi = i // wc * kh, i % wc * kw
            outputs[:, :, hi:(hi + kh), wi:(wi + kw)] = col[:, :, i, :].reshape((m, c, kh, kw))
        return outputs.transpose((0, 2, 3, 1))

    def backward(self, dz):
        m, h, w, c = dz.shape
        kh, kw = self.kernel_size
        arg = self.cache[0]
        dz = arg * dz.transpose(0, 3, 1, 2).reshape((m, c, h * w, 1))
        out_height, out_width = (h - 1) * self.stride + kh, (w - 1) * self.stride + kw
        return self.col2img(col=dz, kh=kh, kw=kw, h=out_height, w=out_width)


class AvgPool2d(Layer):

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs):
        kh, kw = self.kernel_size
        m, xh, xw, channels = inputs.shape
        x = np.transpose(inputs, (0, 3, 1, 2))
        stack = Conv2d.img2col(img=x, kw=kw, kh=kh, stride=self.stride)
        out_height, out_width = (xh - kh) // self.stride + 1, (xw - kw) // self.stride + 1
        outputs = np.mean(stack, axis=3).reshape((m, channels, out_height, out_width)).transpose(0, 2, 3, 1)
        return outputs

    @staticmethod
    def col2img(col, kh, kw, h, w):
        m, c, steps = col.shape[:3]
        outputs = np.zeros((m, c, h, w))
        wc = w // kw
        for i in range(steps):
            hi, wi = i // wc * kh, i % wc * kw
            outputs[:, :, hi:(hi + kh), wi:(wi + kw)] = col[:, :, i, :].reshape((m, c, kh, kw))
        return outputs.transpose((0, 2, 3, 1))

    def backward(self, dz):
        m, h, w, c = dz.shape
        kh, kw = self.kernel_size
        dz = dz.transpose(0, 3, 1, 2).reshape((m, c, h * w, 1)).repeat(kh * kw, axis=3) / (kh * kw)
        out_height, out_width = (h - 1) * self.stride + kh, (w - 1) * self.stride + kw
        return self.col2img(col=dz, kh=kh, kw=kw, h=out_height, w=out_width)


class Flatten(Layer):

    def forward(self, inputs):
        self.shape = inputs.shape
        a = inputs.reshape((self.shape[0], -1))
        return a

    def backward(self, da):
        return da.reshape(self.shape)


class Droupout(Layer):

    def __init__(self, p=0.5):
        self.p = p

    def forward(self, inputs):  # inputs: m * feature_num
        m, feature_num = inputs.shape
        self.drop = np.random.rand(1, feature_num) < self.p
        return inputs * self.drop / self.p

    def backward(self, da):
        return da * self.drop * self.p


class Relu(Layer):

    def forward(self, z):
        self.cache = [z]
        return (z + abs(z)) / 2

    def backward(self, da):
        z = self.cache[0]
        dz = da * (z > 0).astype(int)
        return dz


class Sigmoid(Layer):

    def forward(self, z):
        a = np.exp(z) / (1 + np.exp(z))
        self.cache = [a]
        return a

    def backward(self, da):
        a = self.cache[0]
        dz = da * (1 - a) * a
        return dz


class Tanh(Layer):

    def forward(self, z):
        ez = np.exp(z)
        nez = np.exp(-z)
        a = (ez - nez) / (ez + nez)
        self.cache = [a]
        return a

    def backward(self, da):
        a = self.cache[0]
        return da * (1 - a ** 2)


class LossFunction:

    def __init__(self, nnModel):
        self.nn = nnModel

    def loss_compute(self, inputs, target):
        raise NotImplementedError

    def __call__(self, inputs, target):
        return self.loss_compute(inputs, target)


class CrossEntropyLoss(LossFunction):

    @staticmethod
    def softmax(z):
        ez = np.exp(z)
        y_hat = ez / np.sum(ez, axis=1).reshape(ez.shape[0], 1)
        return y_hat

    def loss_compute(self, inputs, target):
        '''
        target: one-hot array
        '''
        y_hat = self.softmax(inputs)
        grad = y_hat - target
        loss = -np.sum(target * np.log(y_hat)) / target.shape[0]
        return Loss(loss, grad, self.nn)


class MSELoss(LossFunction):

    def loss_compute(self, inputs, target):
        '''
        target: one-hot array
        '''
        grad = 2 * (inputs - target) #/ inputs.shape[0]
        loss = np.mean((inputs - target) ** 2)
        return Loss(loss, grad, self.nn)

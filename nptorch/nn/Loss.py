
class Loss:

    def __init__(self, loss, grad, nnModel):
        self.nn = nnModel
        self.item = loss
        self.grad = grad

    def backward(self):
        self.nn.backward(self.grad)
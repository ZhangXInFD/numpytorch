from nptorch import nn
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def eval(y_test_hat, y_test, nums, flag=False):
    test_hat = (y_test_hat == np.max(y_test_hat, axis = 1).reshape(y_test.shape[0], 1)).astype(int)
    labels = np.array(list(range(nums))).reshape(nums, 1) + 1
    th = np.matmul(test_hat, labels)
    if flag:
        tmp = np.matmul(y_test, labels)
        return np.sum(th == tmp)
    #t = np.matmul(y_test, labels)
    else:
        result = (th == y_test)
        return np.sum(result), np.where(result == 0), th



if __name__ == '__main__':
    data = loadmat('digits.mat')
    x = data['Xtest'] / 255
    x = x.reshape(x.shape[0], 16, 16, 1).transpose(0, 2, 1, 3)
    y = data['ytest']
    model = nn.Sequential([nn.functional.Conv2d((3, 3), in_channels=1, out_channels=16, padding=1),
                            nn.functional.Relu(),
                            nn.functional.MaxPool2d((2, 2), 2),
                            nn.functional.Conv2d((3, 3), in_channels=16, out_channels=32, padding=1),
                            nn.functional.Relu(),
                            nn.functional.MaxPool2d((2, 2), 2),
                            nn.functional.Conv2d((3, 3), in_channels=32, out_channels=64, padding=1),
                            nn.functional.Relu(),
                            nn.functional.MaxPool2d((2, 2), 2),
                            nn.functional.Flatten(),
                            nn.functional.Linear(inputs_dim=256, outputs_dim=128),
                            nn.functional.Relu(),
                            nn.functional.Droupout(),
                            nn.functional.Linear(inputs_dim=128, outputs_dim=10)])
    model.load_state_dict(path='log\\bestmodel\model.pkl')
    m = y.shape[0]
    y_hat = model(x, eval_pattern=True)
    t, f, pred = eval(y_hat, y, 10)
    fi = f[0]
    for i in fi:
        plt.imshow(x[i], cmap='binary')
        plt.title('True label:{}\nModel classify label:{}'.format(y[i][0], pred[i][0]))
        plt.savefig('error{}.png'.format(i))
        plt.show()
    print("Acc on test is {}".format(t / m))
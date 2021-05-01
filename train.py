from nptorch import nn
from nptorch import optim
from tqdm import tqdm
from scipy.io import loadmat
import os
import time
import matplotlib.pyplot as plt
from nptorch.utils.data import Dataset, Dataloader
from nptorch.torchvision.Transform import *
from random import choice

# def mini_batch(x, y=None, batchsize=32, seed=0, method='train'):
#     np.random.seed(seed)
#     m = x.shape[0]
#     permutation = list(np.random.permutation(m))
#     dataiter = []
#     for i in range(0, m, batchsize):
#         if i + batchsize < m:
#             batch_x = x[permutation[i: (i + batchsize)], :]
#             if method == 'train':
#                 batch_y = y[permutation[i: (i + batchsize)], :]
#                 dataiter.append((batch_x, batch_y))
#             else:
#                 dataiter.append(batch_x)
#         else:
#             batch_x = x[permutation[i:], :]
#             if method == 'train':
#                 batch_y = y[permutation[i:], :]
#                 dataiter.append((batch_x, batch_y))
#             else:
#                 dataiter.append(batch_x)
#     return dataiter

class TrainData(Dataset):

    def __init__(self, transforms=[Resize(), Rotation(), Translation()], Augment_p=0.5):
        data = loadmat('digits.mat')
        x = data['X'] / 255
        self.x = x.reshape(x.shape[0], 16, 16, 1).transpose(0, 2, 1, 3)
        y = data['y'].reshape(-1) - 1
        self.y = np.eye(10)[y]
        self.transforms = transforms
        self.p = Augment_p

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        x = self.x[index].copy()
        if self.transforms:
            select = np.random.rand(x.shape[0])
            select = np.where(select <= self.p)[0]
            for i in select:
                t = choice(self.transforms)
                x[i] = t(x[i])
        return (x, self.y[index])


class ValidData(Dataset):

    def __init__(self):
        data = loadmat('digits.mat')
        x = data['Xvalid'] / 255
        self.x = x.reshape(x.shape[0], 16, 16, 1).transpose(0, 2, 1, 3)
        y = data['yvalid']
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

def accuracy(y_test_hat, y_test, nums, flag=False):
    test_hat = (y_test_hat == np.max(y_test_hat, axis = 1).reshape(y_test.shape[0], 1)).astype(int)
    labels = np.array(list(range(nums))).reshape(nums, 1) + 1
    th = np.matmul(test_hat, labels)
    if flag:
        tmp = np.matmul(y_test, labels)
        return np.sum(th == tmp)
    #t = np.matmul(y_test, labels)
    else:
        return np.sum(th == y_test)


def train(model, epoch_num=100, lr=1e-3, train_batchsize=32, seed=521, record=True, plot=True,
          lr_decay=1.0, monitor=True, transform=False):
    #name = model.name
    #print('{} train begin!'.format(name))
    if record:
        path = os.path.join('log', 'baseline{}epoch_num{}lrdecay{}'.format(lr, epoch_num,lr_decay))
        if not os.path.exists(path):
            os.makedirs(path)
    np.random.seed(seed)
    criterion = nn.functional.CrossEntropyLoss(nnModel=model)
    #criterion = nn.functional.MSELoss(nnModel=model)
    optimizer = optim.SGD(model, lr=lr, lr_decay=lr_decay)
    #optimizer = optim.Adam(model)
    costs = []
    traindata = TrainData(transforms=[Resize(), Rotation(), Translation()] if transform else None)
    validdata = ValidData()
    mtrain = traindata.__len__()
    mvalid = validdata.__len__()
    trainloader = Dataloader(traindata, batch_size=train_batchsize)
    validloader = Dataloader(validdata, batch_size=128)
    train_accs = []
    valid_accs = []
    train_time = []
    test_time = []
    for epoch in range(epoch_num):
        tn = 0
        tic = time.time()
        cost = 0
        for x_train, y_train in tqdm(trainloader):
            y_hat = model(x_train)
            loss = criterion(y_hat, y_train)
            costs.append(loss.item)
            cost += loss.item
            tn += accuracy(y_test_hat=y_hat, y_test=y_train, nums=10, flag=True)
            loss.backward()
            optimizer.step()
        train_time.append(time.time() - tic)
        train_accs.append(tn / mtrain)
        tn = 0
        tic = time.time()
        for xv, yv in validloader:
            yv_hat = model(xv, eval_pattern=True)
            tn += accuracy(yv_hat, yv, 10)
        test_time.append(time.time() - tic)
        valid_accs.append(tn / mvalid)
        if monitor:
            print('Epoch:{}\tcost:{}\ttrain acc:{}\tvalid acc:{}'.format(epoch + 1, cost, train_accs[-1], valid_accs[-1]))
        if record and valid_accs[-1] == max(valid_accs):
            model.save_state_dict(os.path.join(path, 'model.pkl'))
    if record:
        accs = np.array([train_accs, valid_accs])
        time_cost = np.array([train_time, test_time])
        np.save(os.path.join(path, 'cost.npy'), costs)
        np.save(os.path.join(path, 'acc.npy'), accs)
        np.save(os.path.join(path, 'time.npy'), time_cost)
    if plot:
        plt.plot(list(range(1, len(costs) + 1)), costs)
        plt.title('Loss Curve')
        plt.ylabel('Loss')
        plt.xlabel('Iter num')
        plt.show()
        plt.plot(list(range(1, len(train_accs) + 1)), 1 - np.array(train_accs), color='blue', label='train')
        plt.plot(list(range(1, len(valid_accs) + 1)), 1 - np.array(valid_accs), color='red', label='valid')
        plt.legend()
        plt.title('Error Curve. Best performance on valid is {}%'.format(max(valid_accs) * 100))
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.show()



if __name__ == '__main__':
    # data = loadmat('digits.mat')
    # x = data['X'] / 255
    # x = x.reshape(x.shape[0], 16, 16, 1).transpose(0, 2, 1, 3)
    # y = data['y'].reshape(-1) - 1
    # xvalid = data['Xvalid'] / 255
    # xvalid = xvalid.reshape(x.shape[0], 16, 16, 1).transpose(0, 2, 1, 3)
    # yvalid = data['yvalid']
    # y = np.eye(10)[y]
    # models = [nn.Sequential([nn.functional.Conv2d((3, 3), in_channels=1, out_channels=4, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.Flatten(),
    #                        nn.functional.Linear(inputs_dim=1024, outputs_dim=256),
    #                        nn.functional.Relu(),
    #                        nn.functional.Linear(inputs_dim=256, outputs_dim=10)]),
    #           nn.Sequential([nn.functional.Conv2d((3, 3), in_channels=1, out_channels=4, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.Flatten(),
    #                        nn.functional.Linear(inputs_dim=1024, outputs_dim=256),
    #                        nn.functional.Relu(),
    #                        nn.functional.Droupout(),
    #                        nn.functional.Linear(inputs_dim=256, outputs_dim=10)]),
    #           nn.Sequential([nn.functional.Conv2d((3, 3), in_channels=1, out_channels=4, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.Conv2d((3, 3), in_channels=4, out_channels=8, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.Flatten(),
    #                        nn.functional.Linear(inputs_dim=2048, outputs_dim=512),
    #                        nn.functional.Relu(),
    #                        nn.functional.Droupout(),
    #                        nn.functional.Linear(inputs_dim=512, outputs_dim=10)]),
    #           nn.Sequential([nn.functional.Conv2d((3, 3), in_channels=1, out_channels=4, padding=1),
    #                          nn.functional.Relu(),
    #                          nn.functional.MaxPool2d((2, 2), 2),
    #                          nn.functional.Conv2d((3, 3), in_channels=4, out_channels=8, padding=1),
    #                          nn.functional.Relu(),
    #                          nn.functional.Flatten(),
    #                          nn.functional.Linear(inputs_dim=512, outputs_dim=256),
    #                          nn.functional.Relu(),
    #                          nn.functional.Droupout(),
    #                          nn.functional.Linear(inputs_dim=256, outputs_dim=10)]),
    #           nn.Sequential([nn.functional.Conv2d((3, 3), in_channels=1, out_channels=8, padding=1),
    #                          nn.functional.Relu(),
    #                          nn.functional.MaxPool2d((2, 2), 2),
    #                          nn.functional.Conv2d((3, 3), in_channels=8, out_channels=32, padding=1),
    #                          nn.functional.Relu(),
    #                          nn.functional.MaxPool2d((2, 2), 2),
    #                          nn.functional.Flatten(),
    #                          nn.functional.Linear(inputs_dim=512, outputs_dim=256),
    #                          nn.functional.Relu(),
    #                          nn.functional.Droupout(),
    #                          nn.functional.Linear(inputs_dim=256, outputs_dim=10)]),
    #           nn.Sequential([nn.functional.Conv2d((3, 3), in_channels=1, out_channels=16, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.MaxPool2d((2, 2), 2),
    #                        nn.functional.Conv2d((3, 3), in_channels=16, out_channels=32, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.MaxPool2d((2, 2), 2),
    #                        nn.functional.Conv2d((3, 3), in_channels=32, out_channels=64, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.MaxPool2d((2, 2), 2),
    #                        nn.functional.Flatten(),
    #                        nn.functional.Linear(inputs_dim=256, outputs_dim=128),
    #                        nn.functional.Relu(),
    #                        nn.functional.Droupout(),
    #                        nn.functional.Linear(inputs_dim=128, outputs_dim=10)]),
    #           nn.Sequential([nn.functional.Conv2d((3, 3), in_channels=1, out_channels=16, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.MaxPool2d((2, 2), 2),
    #                        nn.functional.Conv2d((3, 3), in_channels=16, out_channels=32, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.Conv2d((3, 3), in_channels=32, out_channels=64, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.MaxPool2d((2, 2), 2),
    #                        nn.functional.Conv2d((3, 3), in_channels=64, out_channels=128, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.MaxPool2d((2, 2), 2),
    #                        nn.functional.Flatten(),
    #                        nn.functional.Linear(inputs_dim=512, outputs_dim=128),
    #                        nn.functional.Relu(),
    #                        nn.functional.Droupout(),
    #                        nn.functional.Linear(inputs_dim=128, outputs_dim=10)])]
    # models = list(reversed(models))
    # while models:
    #     model = models.pop()
    #     train(model=model, x=x, y=y, xvalid=xvalid, yvalid=yvalid)
    # models = [nn.Sequential([nn.functional.Conv2d((3, 3), in_channels=1, out_channels=16, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.MaxPool2d((2, 2), 2),
    #                        nn.functional.Conv2d((3, 3), in_channels=16, out_channels=32, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.MaxPool2d((2, 2), 2),
    #                        nn.functional.Conv2d((3, 3), in_channels=32, out_channels=64, padding=1),
    #                        nn.functional.Relu(),
    #                        nn.functional.MaxPool2d((2, 2), 2),
    #                        nn.functional.Flatten(),
    #                        nn.functional.Linear(inputs_dim=256, outputs_dim=128),
    #                        nn.functional.Relu(),
    #                        nn.functional.Droupout(),
    #                        nn.functional.Linear(inputs_dim=128, outputs_dim=10)]),
    #          nn.Sequential([nn.functional.Conv2d((3, 3), in_channels=1, out_channels=16, padding=1),
    #                         nn.functional.Relu(),
    #                         nn.functional.MaxPool2d((2, 2), 2),
    #                         nn.functional.Conv2d((3, 3), in_channels=16, out_channels=32, padding=1),
    #                         nn.functional.Relu(),
    #                         nn.functional.MaxPool2d((2, 2), 2),
    #                         nn.functional.Conv2d((3, 3), in_channels=32, out_channels=64, padding=1),
    #                         nn.functional.Relu(),
    #                         nn.functional.MaxPool2d((2, 2), 2),
    #                         nn.functional.Flatten(),
    #                         nn.functional.Linear(inputs_dim=256, outputs_dim=128),
    #                         nn.functional.Relu(),
    #                         nn.functional.Droupout(),
    #                         nn.functional.Linear(inputs_dim=128, outputs_dim=10)])
    #          ]
    model = nn.Sequential([ nn.functional.Flatten(),
                            nn.functional.Linear(inputs_dim=256, outputs_dim=128),
                            nn.functional.Relu(),
                            nn.functional.Droupout(),
                            nn.functional.Linear(inputs_dim=128, outputs_dim=10),
                            ])
    #model.load_state_dict(path='C:\\Users\ZhangXin\OneDrive\Deep learning\project\project1\project1\log\Fmodellr0.1epoch_num50\model.pkl')
    # model = models.pop()
    # #train(model=model, epoch_num=200, lr=1e-1, seed=123, train_batchsize=64, lr_decay=0.999, record=True)
    # train(model=model, epoch_num=200, lr=1e-1, seed=123, train_batchsize=64, lr_decay=0.9996, record=True)
    # model = models.pop()
    # train(model=model, epoch_num=200, lr=1e-1, seed=123, train_batchsize=64, lr_decay=0.999, record=True)
    train(model=model, epoch_num=200, lr=1e-1, seed=123, train_batchsize=64, lr_decay=1, record=False)


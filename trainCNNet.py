'''
FileName: trainCNNet.py
Author: Chuncheng
Version: V0.0
Purpose: Train the CNNet
'''

# %%
import keras
import torch
import numpy as np

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from Package import logger
from Package.MnistDataset import DataSet
from Package.CNN import cnn_summary, optim, F

# %%
# Converting Tools


def numpy2cuda(x):
    ''' Convert numpy matrix [x] to cuda tensor '''
    return torch.from_numpy(x).cuda()


def cuda2numpy(x):
    ''' Convert cuda tensor [x] to numpy matrix '''
    return x.cpu().numpy()


# Converting Examples
print('--------------------------------------------')
x = np.random.randint(30 + np.zeros((600, 30, 40))).astype(np.float32)
print(x.shape)
print(x.dtype)
print(numpy2cuda(x).dtype)
print(cuda2numpy(numpy2cuda(x)).dtype)

# %%
# Load Data
tmp = keras.datasets.mnist.load_data()
(X_train, y_train), (X_test, y_test) = tmp
logger.debug('Data Info is {}'.format((X_train.shape, y_train.shape,
             X_test.shape, X_test.shape, X_train.dtype, y_train.dtype)))

# Load Noisy Data
tmp = DataSet()
X_noisy, y_noisy = tmp.load_pics('line_deletion')

# %%
# Setup
optimizer_kwargs = dict(lr=0.01, momentum=0.1)
net = cnn_summary()
optimizer = optim.SGD(net.parameters(), **optimizer_kwargs)

# Independent Testing Set
X_test_cuda = numpy2cuda(X_test.astype(np.float32)[:, np.newaxis])
X_noisy_cuda = numpy2cuda(X_noisy.astype(np.float32)[:, np.newaxis])


class Trainer(object):
    ''' CNNet Trainer Object '''

    def __init__(self, net=net, optimizer=optimizer):
        '''
        Method: __init__

        Initialize CNNet Training Object

        Args:
        - @self, net=net, optimizer=optimizer

        '''
        self.net = net
        self.optimizer = optimizer
        pass

    def train(self, epoch_idx=0):
        '''
        Method: train

        Training Epoch

        Args:
        - @self, epoch_idx=0

        '''
        n_splits = 100
        log_interval = 20
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        batch_idx = 0
        for _, idx2 in skf.split(X_train, y_train):
            # logger.debug('Selecting {} samples for the Epoch'.format(len(idx2)))
            X = X_train[idx2].astype(np.float32)
            y = y_train[idx2]

            self.optimizer.zero_grad()
            output = self.net(numpy2cuda(X[:, np.newaxis]))
            loss = F.nll_loss(output, numpy2cuda(y).long())
            loss.backward()
            self.optimizer.step()

            if batch_idx % log_interval == 0:
                p1 = cuda2numpy(output.data.max(1)[1])
                acc1 = metrics.accuracy_score(y, p1)

                output2 = self.net(X_test_cuda, training=False)
                p2 = cuda2numpy(output2.data.max(1)[1])
                acc2 = metrics.accuracy_score(y_test, p2)

                output3 = self.net(X_noisy_cuda, training=False)
                p3 = cuda2numpy(output3.data.max(1)[1])
                acc3 = metrics.accuracy_score(y_noisy, p3)

                print('Train Epoch: {} [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}, {:.4f}, {:.4f}'.format(
                    epoch_idx,
                    (batch_idx+1) * len(y),
                    len(y_train),
                    loss.item(),
                    acc1,
                    acc2,
                    acc3
                ))

            batch_idx += 1

        pass


trainer = Trainer()

# %%
for epoch_idx in range(20):
    trainer.train(epoch_idx)

# %%

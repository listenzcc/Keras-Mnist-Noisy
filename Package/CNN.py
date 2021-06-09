'''
FileName: CNN.py
Author: Chuncheng
Version: V0.0
Purpose: CNN Architecture Object
'''

# %%
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary

from . import logger


# %%
class CNNet(nn.Module):
    ''' Convolution Neural Network for Mnist Dataset '''

    def __init__(self):
        '''
        Method:__init__

        Initialization of the Object.

        Args:
        - @self

        '''

        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        logger.info('CNNet is Initialized'.format())

        pass

    def forward(self, x, training=True):
        '''
        Method:forward

        Forward method

        Args:
        - @self
        - @x: The Input tensor;
        - @training: The Option of Training State, default value is True.

        Outputs:
        - The Output tensor.

        '''
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


# %%
def cnn_summary(show_summary=True):
    '''
    Method:cnn_summary

    Build CNNet Object with Cuda Supporting and Summary it

    Args:
    - show_summary: If show summary of the CNN net, default value is True.

    Outputs:
    - The CNNet Object

    '''
    net = CNNet().cuda()

    if show_summary:
        summary(net, input_size=(1, 28, 28))

    logger.debug('CNNet instance is generated'.format())
    return net


# %%
if __name__ == '__main__':
    cnn_summary(show_summary=True)


# %%

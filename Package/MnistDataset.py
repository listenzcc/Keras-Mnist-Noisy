'''
FileName: MnistDataset.py
Author: Chuncheng
Version: V0.0
Purpose: Load and Put MnistDataset
'''

import os
import numpy as np

from tqdm.auto import tqdm

from PIL import Image

from . import logger, CFG


# ------------------------------------------------------------------------------
# Utils

def read_pic(path, gray=True):
    '''
    Method:read_pic

    Read Picture into Numpy Array

    Args:
    - @path: The path of the picture;
    - @gray: Whether to assume the picture is in gray colors, default value is True.

    Outputs:
    - The 3-D array of the RGB[A] matrix;
    - If the gray option is True, the output will be the 2-D array of the Gray matrix.

    '''

    assert(os.path.isfile(path))
    img = Image.open(path)

    mat = np.array(img)

    if not gray:
        return mat
    else:
        return mat[:, :, 0]


# ------------------------------------------------------------------------------
# Dataset Object


class DataSet(object):
    ''' The Mnist DataSet
    '''

    def __init__(self, noisy_folder=CFG['Default']['noisyFolder']):
        '''
        Method:__init__

        Init the DataSet of Mnist

        Args:
        - @noisy_folder: The folder of the noisy .png pictures, has default value.

        '''

        self.noisy_folder = noisy_folder
        logger.debug(f'DataSet uses the folder of "{noisy_folder}"')

        self.get_noises()

        logger.info(f'DataSet initialized {noisy_folder}')

        pass

    def get_noises(self):
        '''
        Method:get_noises

        Get the noises from the noisy_folder.

        Args:
        - @self

        '''

        self.noises = [e for e in os.listdir(self.noisy_folder)
                       if not e.startswith('.')]
        logger.debug('Detected noises of {}'.format(self.noises))

    def load_pics(self, noise_name=None):
        '''
        Method:load_pics

        Load the pictures of the noise_name.

        Args:
        - @noise_name: The name of the noise, default value is None.

        Outputs:
        - @X: The picture matrix;
        - @y: The label vector.

        '''

        if not noise_name in self.noises:
            logger.debug('The noise {} is incorrect, changing it into {}'.format(
                noise_name, self.noises[0]))
            noise_name = self.noises[0]
        logger.info('Using the noise of {}'.format(noise_name))

        folder = os.path.join(self.noisy_folder, noise_name, '0')

        X = []
        y = []
        for num in tqdm(range(10)):
            num_folder = os.path.join(folder, str(num))
            for name in os.listdir(num_folder):
                path = os.path.join(num_folder, name)
                X.append(read_pic(path))
                y.append(num)
            pass

        X = np.array(X)
        y = np.array(y)

        logger.debug('The X is {}, y is {}'.format(X.shape, y.shape))

        return X, y

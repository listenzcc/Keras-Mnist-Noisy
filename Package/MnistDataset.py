'''
FileName: MnistDataset.py
Author: Chuncheng
Version: V0.0
Purpose: Load and Put MnistDataset
'''

import os
import numpy as np

from PIL import Image

from . import logger, Cfg


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
    - The 3-D array of the RGB[A] matrix.
    -

    '''

    assert(os.path.isfile(path))
    img = Image.open(path)
    return np.array(img)


# ------------------------------------------------------------------------------
# Dataset Object


class DataSet(object):
    ''' The Mnist DataSet
    '''

    def __init__(self, noisy_folder=Cfg['Default']['noisyFolder']):
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

        Outputs:
        - @

        '''

        self.noises = [e for e in os.listdir(self.noisy_folder)
                       if not e.startswith('.')]
        logger.debug('Detected noises of {}'.format(self.noises))

        pass

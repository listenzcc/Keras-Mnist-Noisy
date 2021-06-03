'''
FileName: script.py
Author: Chuncheng
Version: V0.0
Purpose: Python script of keras demo on mnist dataset.
'''

# %%
from PIL import Image
import numpy as np
import keras
import os
from Package.MnistDataset import DataSet

data = DataSet()

# %%

# %%


# %%
mnist = keras.datasets.mnist.load_data()
mnist

# %%
mnist[0][0].shape

# %%
png_path = os.path.join(os.environ['HOME'],
                        r'Documents/science_rcn/noisy_tests/bg_noise/0/2',
                        'test_996.png')

os.path.isfile(png_path)

# %%
img = Image.open(png_path)
dir(img)

# %%
img.width
# %%
np.array(img).shape
# %%

'''
FileName: script.py
Author: Chuncheng
Version: V0.0
Purpose: Python script of keras demo on mnist dataset.
'''

# %%
import plotly.express as px
import os
import keras
import numpy as np
from PIL import Image
from Package.MnistDataset import DataSet

# %%
data = DataSet()
Xd, yd = data.load_pics()
Xd.shape, yd.shape

# %%
mnist = keras.datasets.mnist.load_data()
(Xm, ym), (_, _) = mnist
Xm.shape, ym.shape

# %%


def plot(X, y):
    fig = px.imshow(X[0])
    fig.update_layout(title=str(y[0]))
    return fig

# %%

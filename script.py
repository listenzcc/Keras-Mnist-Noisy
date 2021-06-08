'''
FileName: script.py
Author: Chuncheng
Version: V0.0
Purpose: Python script of keras demo on mnist dataset.
'''

# %%
import keras
import random
import numpy as np
import plotly.express as px

from PIL import Image
from Package.MnistDataset import DataSet

# %%
data = DataSet()
Xd, yd = data.load_pics('line_deletion')
Xd.shape, yd.shape, Xd.dtype

# %%
mnist = keras.datasets.mnist.load_data()
(Xm, ym), (_, _) = mnist
Xm.shape, ym.shape, Xm.dtype

# %%

height = 500
width = 500
coloraxis = dict(colorscale=[[0.0, '#111'],
                             [1.0, '#EEE']])

fig_kwargs = dict(height=height,
                  width=width,
                  coloraxis=coloraxis)


def plot(X, y, idx=-1, kwargs=fig_kwargs):
    '''
    Method:plot

    Plot the Mnist Plotting as the [idx].

    Args:
    - @X: The Image Matrix;
    - @y: The Label Vector;
    - @idx=-1: The Index of interest, -1 means random selection;
    - @kwargs=fig_kwargs: The keyword arguments of the Fig:

    Outputs:
    - The figure of the Mnist.

    '''

    if idx == -1:
        idx = random.randint(0, X.shape[0])

    mat = X[idx]
    title = str(y[idx])

    kwargs = dict(kwargs,
                  title=title)

    fig = px.imshow(mat)
    fig.update_layout(**kwargs)

    return fig


# %%
fig = plot(Xm, ym)
fig.show()

fig = plot(Xd, yd)
fig.show()

# %%

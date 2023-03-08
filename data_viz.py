# -*- coding: utf-8 -*-

from matplotlib import colors
import matplotlib.pyplot as plt


def gen_plot(inputs):
    """
    Parameters
    ----------
    inputs : Torch Tensor BCHW within [0,1]
        MeteoNet data are greyscale so channel input is used as temporal channel to store images of the 12-sequence. 

    Returns
    -------
    buf : Buffer缓冲区
        Buffer to temporarily save the figure.

    """
    nb_batch = inputs.shape[0]
    nb_channels = inputs.shape[1]
    cmap_BW = colors.ListedColormap(['red','black', 'white'])
    cmap_col = colors.ListedColormap(['indianred', 'mediumseagreen' ,'green', 'red'])
    legend = "-1-Pale red : pred=0,target=1 / 0-SeaGreen : pred=0,target=0 / 1-Green : pred=1,target=1 / 2-Red : pred=1,target=0"
    fig,axes = plt.subplots(nb_batch,nb_channels,figsize=(15,nb_batch*4))
    n=1
    #label = ["Target","Prediction","Persistance","2*Pred-Tgt","2*Pers-Tgt"]
    label = ["Target","Prediction","Pred-Tgt"]
    for i in range(nb_batch):
        for j in range(nb_channels):
            cmap=cmap_BW if (i<2) else cmap_col
            #cmap=cmap_col
            bounds =  [-1,-0.5,0.5,1] if (i<2) else [-1,-0.5,0.5,1.5,2]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            ax = axes[i,j] if nb_batch>1 else axes[j]
            ax.set(aspect='equal')
            pl=ax.pcolormesh(inputs[i,j].to("cpu"),cmap=cmap, norm=norm)
            if j==0:
                ax.set_ylabel(label[i])
            n += 1
                
    cbar = fig.colorbar(pl,ax=axes.ravel().tolist(),cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 
                    orientation= 'vertical').set_label(legend)
    return fig

from matplotlib import pyplot as plt
import numpy as np
from utils import kullback_leibler_divergence as kld


def select_layers(d,layer_names):
    output = {}
    for name in layer_names:
        if name in d.keys():
            output[name] = d[name]
    return d

def new_boxplot_kl_frames(layer_to_vector_dict, filename = '', 
    layer_names = []):
    kl_type = 'normal'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    axis = [ax1,ax2]
    for i,key in enumerate(layer_to_vector_dict.keys()):
        d = select_layers(layer_to_vector_dict[key],layer_names)
        v = list(layer_to_vector_dict[key].values())
        if type(v[0]) != float: 
            v = kld.frame_lists_to_kl_vector(v,kl_type)
        else:print('not using random_kl setting, kl in vector is used')
        ticklabels = list(layer_to_vector_dict[key].keys())
        if key == 'ctc': 
            v = [np.array([])] + v
            ticklabels = [''] + ticklabels
        else:
            ticklabels[ticklabels.index('cnn_features')] = 'cnn'
        axis[i].boxplot(v)
        axis[i].title.set_text(key)
        axis[i].xaxis.set_ticklabels(ticklabels)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    axis[-1].set_xlabel('wav2vec 2.0 layer')
    axis[-1].set_ylabel('kullback-leibler divergence')
    if filename:
        fig.savefig(filename)
    return fig

def boxplot_kl_frames(layer_to_vector_dict, name = '', filename = '',
    kl_type = 'normal'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    axis = [ax1,ax2]
    for i,key in enumerate(layer_to_vector_dict.keys()):
        v = list(layer_to_vector_dict[key].values())
        if type(v[0]) != float: 
            v = kld.frame_lists_to_kl_vector(v,kl_type)
        else:print('not using random_kl setting, kl in vector is used')
        ticklabels = list(layer_to_vector_dict[key].keys())
        if key == 'ctc': 
            v = [np.array([])] + v
            ticklabels = [''] + ticklabels
        else:
            ticklabels[ticklabels.index('cnn_features')] = 'cnn'
        axis[i].boxplot(v)
        axis[i].title.set_text(key)
        axis[i].xaxis.set_ticklabels(ticklabels)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    axis[-1].set_xlabel('wav2vec 2.0 layer')
    axis[-1].set_ylabel('kullback-leibler divergence')
    if not name:
        name = 'wav2vec 2.0 hidden states based MLP phoneme classifier'
        name += ' kullback-leibler divergence'
    fig.suptitle(name)
    if filename:
        fig.savefig(filename)
    return fig

from matplotlib import pyplot as plt
import numpy as np
from utils import kullback_leibler_divergence as kld
from matplotlib.lines import Line2D

def custom_lines(colors = ['pink','yellow','purple']):
    custom_lines = []
    for color in colors:
        l = Line2D([0],[0],color=color, lw=8)
        custom_lines.append(l)
    return custom_lines

def add_legend(axis, lines = None):
    if not lines: 
        lines = custom_lines(boxplot_colors()[:3])
    axis.legend(lines, ['BPC','random','difference'])


def select_layers(d,layer_names):
    output = {}
    for name in layer_names:
        if name in d.keys():
            output[name] = d[name]
    return output

def make_ticklabels(layer_names= [], model_type = ''):
    if not layer_names: layer_names = ['cnn_features',1,12,24]
    o = []
    for name in layer_names:
        for n in ['BPC','random','difference']:
            if name == 'cnn_features': name = 'cnn'
            o.append(name)
    if model_type == 'ctc': ['','',''] + o
    return o

def make_short_ticklabels(layer_names=[], model_type = ''):
    if not layer_names: layer_names = ['cnn_features',1,12,24]
    if model_type == 'ctc': 
        layer_names = [''] + layer_names
    else: layer_names[0] = 'cnn'
    return layer_names

        
def make_tick_locations():
    return [2,5,8,11]

def vertical_lines():
    return [3.5,6.5,9.5]

def boxplot_colors():
    return ['pink','yellow','purple']*4


def new_boxplot_kl_frames(layer_to_frame_dict, filename = '', 
    layer_names = []):
    kl_type = 'normal'
    if not layer_names: layer_names = ['cnn_features',1,12,24]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    axis = [ax1,ax2]
    # ax1.set_facecolor('ivory')
    # ax2.set_facecolor('ivory')
    for i,key in enumerate(layer_to_frame_dict.keys()):
        d = select_layers(layer_to_frame_dict[key],layer_names)
        v = list(d.values())
        bpc = kld.frame_lists_to_kl_vector(v,'normal')
        random= kld.frame_lists_to_kl_vector(v,'random')
        diff= kld.frame_lists_to_kl_vector(v,'diff')
        ticklabels = make_short_ticklabels(list(d.keys()), key)
        ticks = make_tick_locations()
        v = []
        for bpc_l,random_l,diff_l in zip(bpc,random,diff):
            v.extend([bpc_l,random_l,diff_l])
        if key == 'ctc': 
            for _ in range(3):
                v = [np.array([])] + v
        print(ticklabels,len(v))
        fpd= {'marker':'o','alpha':0.01,'markersize':2}
        bp =axis[i].boxplot(v,patch_artist = True, flierprops=fpd)
        [x.set_facecolor(c) for x,c in zip(bp['boxes'],boxplot_colors())]
        [x.set_markeredgecolor(c) for x,c in zip(bp['fliers'],boxplot_colors())]
        [x.set_markerfacecolor(c) for x,c in zip(bp['fliers'],boxplot_colors())]
        print(bp['boxes'],'<---')
        axis[i].title.set_text(key)
        axis[i].xaxis.set_ticks(ticks)
        axis[i].xaxis.set_ticklabels(ticklabels)
        for x in vertical_lines():
            axis[i].axvline(x,linestyle = '--', color = 'grey', alpha = .3)
        axis[i].axhline(0, color = 'grey', alpha = .3, linewidth = 2)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    axis[-1].set_xlabel('Wav2vec 2.0 layer')
    axis[-1].set_ylabel('Kullback-Leibler divergence')
    add_legend(ax2)
    if filename:
        fig.savefig(filename)
    return fig, bp, ax2

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

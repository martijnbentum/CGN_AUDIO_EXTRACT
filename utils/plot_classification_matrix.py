import matplotlib 
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy
from utils import phonemes


def plot_matrix(matrix, vocab, title = '',convert = True):
    if convert:
        matrix, vocab = reorder_and_convert(matrix, vocab)
    ytick_values = list(vocab.values())
    ytick_labels = list(vocab.keys())
    xtick_values = list(range(matrix.shape[0]))
    xtick_labels = [list(vocab.keys())[i] for i in numpy.argmax(matrix,1)] 
    plt.clf()
    im = plt.matshow(matrix.transpose())
    plt.title(title)
    plt.yticks(ticks = ytick_values,labels= ytick_labels, fontsize = 7)
    ax = plt.gca()
    ax.set_xticks(ticks= xtick_values, labels=xtick_labels, fontsize = 5)
    plt.tick_params(axis='x',which='both',top=False, bottom = False)
    # ax.set_xticks([])
    ax.secondary_xaxis('bottom', functions = (to_seconds, to_index))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad=0.05)
    plt.colorbar(im, cax= cax)
    plt.show()


def reorder_and_convert(matrix, vocab):
    ipa_vocab = phonemes.convert_simple_sampa_vocab_to_ipa_symbols(vocab)
    d,i = phonemes.reorder_simple_ipa_vocab_to_phoneme_classes(ipa_vocab)
    m = matrix[:,i]
    return m, d


def to_index(seconds):
    return seconds * 1000 / 20

def to_seconds(index):
    return index * 20 / 1000

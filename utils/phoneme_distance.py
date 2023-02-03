import panphon
from . import phonemes
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle

ipa = phonemes.Ipa()
f_feature_panphan_distance_dict = '../feature_panphon_distance_dict.pickle'

def _average_vector_list(vectors):
    vectors = np.array([np.array(line[:22]) for line in vectors])
    return vectors

def phoneme_to_vector(phoneme):
    if phoneme == 'g': phoneme = 'ɡ'
    ft = panphon.FeatureTable()
    # excluding last to features, see ft.names
    p = ft.word_to_vector_list(phoneme, numeric=True)
    if len(p)> 1: p = _average_vector_list(p)
    else: p = np.array(p[0][:22])
    return p

def compute_distance(phoneme1, phoneme2):
    p1 = phoneme_to_vector(phoneme1)
    p2 = phoneme_to_vector(phoneme2)
    if len(p1.shape) == len(p2.shape) == 1:
        return np.linalg.norm(p1 - p2)
    distance = []
    if len(p1.shape) > 1 and len(p2.shape) > 1:
        for row_p1 in p1:
            for row_p2 in p2:
                distance.append( np.linalg.norm(row_p1 - row_p2) )
        return np.mean(distance)
    elif len(p1.shape) > 1: 
        for row_p1 in p1:
            distance.append( np.linalg.norm(row_p1 - p2) )
        return np.mean(distance)
    elif len(p2.shape) > 1:
        for row_p2 in p2:
            distance.append( np.linalg.norm(p1 - row_p2) )
        return np.mean(distance)


def compute_distance_set(phoneme, phoneme_set, d = None):
    if not d: d = {phoneme:{}}
    else: d[phoneme] = {}
    phoneme_is_vowel = phoneme in ipa.vowels
    for other_phoneme in phoneme_set:
        other_phoneme_is_vowel = other_phoneme in ipa.vowels
        if phoneme_is_vowel != other_phoneme_is_vowel: continue
        distance = compute_distance(phoneme, other_phoneme)
        d[phoneme][other_phoneme] = distance
    d[phoneme] = sort_dictionary_by_value(d[phoneme])
    return d 
        
def compute_all_sets(phoneme_set, save = False):
    n_phonemes = len(phoneme_set)
    d ={} 
    for i in range(n_phonemes):
        p_set = phoneme_set[:]
        phoneme = p_set.pop(i)
        d = compute_distance_set(phoneme, p_set, d)
    save_distance_dictionary(d, f_feature_panphan_distance_dict)
    return d
        
def sort_dictionary_by_value(d):
    o = {k: v for k, v in sorted(d.items(), key=lambda x: x[1])}
    return o


def plot_distance(phoneme, d):
    plt.ion()
    plt.clf()
    plt.ylim((0,7))
    distances = list(d[phoneme].values())
    labels = list(d[phoneme].keys())
    x_indices = list(range(len(distances)))
    plt.plot(distances)
    plt.xticks(x_indices, labels)
    plt.title(phoneme)
    plt.xlabel('phonemes')
    plt.ylabel('distance to ' + phoneme)
    plt.grid() 
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('../phoneme_distance_plots/'+phoneme+'.png')

def plot_all_phoneme_distances(d):
    for phoneme in d.keys():
        plot_distance(phoneme, d)

    

def save_distance_dictionary(d, filename):
    with open(filename,'wb') as fout:
        pickle.dump(d,fout)

def load_distance_dictionary(filename = None):
    if filename == None: filename = '../feature_panphon_distance_dict.pickle'
    with open(filename,'rb') as fin:
        d = pickle.load(fin)
    return d


feature_names = [
    'syl',
    'son',
    'cons',
    'cont',
    'delrel',
    'lat',
    'nas',
    'strid',
    'voi',
    'sg',
    'cg',
    'ant',
    'cor',
    'distr',
    'lab',
    'hi',
    'lo',
    'back',
    'round',
    'velaric',
    'tense',
    'long']

# 0 - syllabic
# 1 - sonorant
# 2 - consonantal
# 3 - continuant
# 4 - delayed release
# 5 - lateral
# 6 - nasal
# 7 - strident
# 8 - voice
# 9 - spread glottis
# 10 - constricted glottis
# 11 - anterior
# 12 - coronal
# 13 - distributed
# 14 - labial
# 15 - high (vowel/consonant, not tone)
# 16 - low (vowel/consonant, not tone)
# 17 - back
# 18 - round
# 19 - tense
# 20 - long

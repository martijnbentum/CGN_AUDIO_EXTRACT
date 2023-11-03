import glob
import json
from utils import locations
import numpy as np
import os
import pickle
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

layers = ['cnn',1,6,12,18,21,24]
sections = ['vowel', 'syllable', 'word']

def train_classifier(stress_info, name , layer, section, overwrite = False,
    random_gt = False, occlusion_type = None):
    if name: name = '_' + name
    if random_gt: name += '-random-gt'
    if occlusion_type: name += '-occlusion-' + occlusion_type
    f=locations.stress_perceptron_dir + 'clf' + name + '_' + section   
    f+= '_' + str(layer) + '.pickle'
    if os.path.isfile(f) and not overwrite:
        print(f, 'already exists, skipping')
        return
    print('starting on',f)
    X, y = stress_info.xy(layer = layer, section = section, 
        random_gt = random_gt, occlusion_type = occlusion_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        stratify=y,random_state=1)
    clf=MLPClassifier(random_state=1,max_iter=300)
    clf.fit(X_train, y_train)
    hyp = clf.predict(X_test)
    save_performance(y_test, hyp, name, layer, section)
    with open(f, 'wb') as fout:
        pickle.dump(clf,fout)

def train_classifiers(stress_info, name = '', layers = layers, 
    sections = sections, random_gt = False, occlusion_type = None):
    '''train mlp classifiers based on the data structure hidden_states.'''
    for layer in layers:
        for section in sections:
            train_classifier(stress_info, name, layer, section, 
                random_gt = random_gt, occlusion_type = occlusion_type)

def save_performance(gt, hyp, name, layer, section):
    d = {}
    d['mcc'] = round(matthews_corrcoef(gt, hyp), 3)
    d['accuracy'] = round(accuracy_score(gt, hyp), 3)
    d['report'] = classification_report(gt, hyp)
    print(name, layer, section)
    for k,v in d.items():
        print(k,v)
    print('---')
    f = locations.stress_perceptron_dir + 'score' + name 
    f +=  '_' + str(layer)+'_'+ section + '.json'
    with open(f, 'w') as fout:
        json.dump(d, fout)
    return d


class Perceptron:
    '''class to hold layer specific classifiers.'''
    def __init__(self, filename = None, layers = layers):
        self.layers = layers
        self.filename = filename
        c = [load_perceptron(l,small,ctc, filename) for l in layers]
        self.classifiers = c


def score_filename_to_layer_section(f):
    layer = f.split('_')[-2]
    section = f.split('_')[-1].split('.')[0]
    return layer, section


def get_scores(name, layer = '*', section = '*', occlusion = False):
    f = locations.stress_perceptron_dir + 'score_' + name 
    if occlusion: f += '-occlusion*'
    f +=  '_' + str(layer) +'_'+ section + '.json'
    fn = glob.glob(f)
    output = {}
    for f in fn:
        print(f)
        layer, section = score_filename_to_layer_section(f)
        with open(f, 'r') as fin:
            d = json.load(fin)
        output[layer,section] = d
    return output

def show_scores(name, section):
    f = locations.stress_perceptron_dir + 'score_' + name 
    f +=  '_*_'+ section + '.json'
    fn = glob.glob(f)
    for f in fn:
        print(f)
        with open(f, 'r') as fin:
            d = json.load(fin)
        print('mcc', d['mcc'])
        print('---')

def plot_scores(name = 'mald-variable-stress-small-pretrained', 
    occlusion =False):
    scores= get_scores(name, occlusion = occlusion)
    print(scores.keys())
    mcc_vowel= [scores[str(layer),'vowel']['mcc'] for layer in layers]
    mcc_syllable= [scores[str(layer),'syllable']['mcc'] for layer in layers]
    if not occlusion:
        mcc_word = [scores[str(layer),'word']['mcc'] for layer in layers]
    plt.clf()
    plt.ylim(-0.1,1)
    plt.plot(mcc_vowel, label = 'vowel')
    plt.plot(mcc_syllable, label = 'syllable')
    if not occlusion:
        plt.plot(mcc_word, label = 'word')
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.xticks(range(len(layers)), layers)
    plt.xlabel('wav2vec 2.0 layer')
    plt.ylabel('matthews correlation coefficient')
    plt.show()

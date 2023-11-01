from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import json
from utils import locations
import numpy as np
import os
import pickle

layers = ['cnn',1,6,12,18,21,24]
sections = ['vowel', 'syllable', 'word']

def train_classifier(stress_info, name , layer, section, overwrite = False):
    if name: name = '_' + name
    f=locations.stress_perceptron_dir + 'clf' + name + '_' + section   
    f+= '_' + str(layer) + '.pickle'
    if os.path.isfile(f) and not overwrite:
        print(f, 'already exists, skipping')
        return
    print('starting on',f)
    X, y = stress_info.xy(layer = layer, section = section)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        stratify=y,random_state=1)
    clf=MLPClassifier(random_state=1,max_iter=300)
    clf.fit(X_train, y_train)
    hyp = clf.predict(X_test)
    save_performance(y_test, hyp, name, layer, section)
    with open(f, 'wb') as fout:
        pickle.dump(clf,fout)

def train_classifiers(stress_info, name = '', layers = layers, 
    sections = sections):
    '''train mlp classifiers based on the data structure hidden_states.'''
    for layer in layers:
        for section in sections:
            train_classifier(stress_info, name, layer, section)

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

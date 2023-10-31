from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import json
import numpy as np
from utils import locations

layers = ['cnn',1,6,12,18,21,24]

def train_classifiers(stress_info, name = '', layers = layers):
    '''train mlp classifiers based on the data structure hidden_states.'''
    if name: name = '_' + name
    for section in ['vowel','syllable','word']:
        for layer in layers:
            f=locations.stress_perceptron_dir + 'clf' + name + '_' + section   
            f+= '_' + str(layer) + '.pickle'
            if os.path.isfile(f):
                print(f, 'already exists, skipping')
                continue
            print('starting on',f)
            X, y = hidden_states.to_dataset(layer)
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                stratify=y,random_state=1)
            clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
            hyp = clf.predict(X_test)
            save_performance(y_test, hyp, name, layer, section)
            with open(f, 'wb') as fout:
                pickle.dump(clf,fout)

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

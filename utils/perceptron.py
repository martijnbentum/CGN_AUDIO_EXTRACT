from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import glob
import os
import pickle
from utils import locations

def save_score(score, name, layer, nfiles):
    f = locations.perceptron_dir + 'score' + name + '_' + str(layer)+'_'+nfiles
    with open(f, 'w') as fout:
        fout.write(str(score))

def train_classifiers(hidden_states, name = ''):
    if name: name = '_' + name
    layers = hidden_states.layer_dict.keys()
    nfiles =len(hidden_states.filenames)
    if nfiles > 0: nfiles = str(nfiles)
    else: nfiles = ''
    for layer in layers:
        f=locations.perceptron_dir + 'clf' + name + '_' + str(layer) 
        f+= '_' + nfiles +'.pickle'
        if os.path.isfile(f):
            print(f, 'already exists, skipping')
            continue
        print('starting on',f)
        X, y = hidden_states.to_dataset(layer)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
            stratify=y,random_state=1)
        clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name, layer, score, nfiles)
        save_score(score, name, layer, nfiles)
        with open(f, 'wb') as fout:
            pickle.dump(clf,fout)

def show_scores():
    fn = glob.glob(locations.perceptron_dir + 'score*')
    for f in fn:
        name = f.split('/')[-1]
        print(name.ljust(35), round(float(open(f).read()),2))
    
def load_perceptron(layer = None, small = True, ctc = None, name = None):
    if layer == None and small == False: layer = 19
    elif layer == None and small == True: layer = 18
    if ctc == None and small == False: ctc = False
    if ctc == None and small == True: ctc = True
    size = 'small' if small else 'big'
    model_type = 'ctc' if ctc else 'pretrained'
    if name == None:
        filename = locations.perceptron_dir + 'clf_' + size + '_' + model_type
        filename += '_' + str(layer) + '_'
        fn = glob.glob(filename +'*.pickle')
        print(filename,fn)
        biggest = 0
        for f in fn:
            n_files = int(f.split('_')[-1].split('.')[0])
            if n_files > biggest: name = f
    if not name or not os.path.isfile(name):
        print('could not find perceptron model with name:', name)
        return
    print('loading perceptron from file:',name)
    with open(name, 'rb') as fin:
        classifier = pickle.load(fin)
    return classifier


def predict_label_hidden_state(clf,hs,layer):
    X = hs.to_dataset(layer)
    return clf.predict_proba(X)
    
class Perceptron:
    def __init__(self, small = True, ctc = None, filename = None, layers = []):
        if not layers: 
            if small: layers = [1,6,12,18,21,24]
            if not ctc: layers = ['cnn_features'] + layers
            else: layers = [1,10,19,28,37,46]
        self.layers = layers
        self.ctc = ctc
        self.filename = filename
        c = [load_perceptron(l,small,ctc, filename) for l in layers]
        self.classifiers = c
    



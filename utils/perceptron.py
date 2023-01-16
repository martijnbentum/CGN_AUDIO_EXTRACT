from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

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
        X, y = hidden_states.to_dataset(layer)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
            stratify=y,random_state=1)
        clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name, layer, score, nfiles)
        save_score(score, name, layer, nfiles)
        f=locations.perceptron_dir + 'clf' + name + '_' + str(layer) 
        f+= '_' + nfiles +'.pickle'
        with open(f, 'wb') as fout:
            pickle.dump(clf,fout)



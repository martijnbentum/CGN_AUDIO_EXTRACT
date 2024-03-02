from utils import locations
from . import stress_info
from matplotlib import pyplot as plt
import os
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

layers = ['cnn',1,6,12,18,21,24]
mald_vowels = 'ɛ ʊ ɪ ɔ i æ u ʌ eɪ aʊ ɔɪ oʊ ai ɑ ɝ'.split(' ')

def get_all_mald_vowels(n = None):
    info = stress_info.Info(dataset_name = 'mald_all')
    if n: syllables = info.syllables[:n]
    else: syllables = info.syllables
    vowels = [syllable.vowel_ipa for syllable in syllables]
    return vowels


def make_vowel_indices_dict(n = None):
    vowels = get_all_mald_vowels(n = n)
    output = {}
    for index, vowel in enumerate(vowels):
        if vowel not in output.keys(): output[vowel] = []
        output[vowel].append(index)
    return output

def make_leave_one_in_train_test_sets(X, y, vowel = 'ɛ', n = None):
    vowel_indices = make_vowel_indices_dict(n = n)
    leave_one_in_indices = vowel_indices[vowel]
    test_indices = [i for i in range(len(X)) if i not in leave_one_in_indices]
    X_train = X[leave_one_in_indices]
    X_test = X[test_indices]
    y_train = y[leave_one_in_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

def make_leave_one_out_train_test_sets(X, y, vowel = 'ɛ', n = None):
    vowel_indices = make_vowel_indices_dict(n = n)
    leave_one_out_indices = vowel_indices[vowel]
    train_indices = [i for i in range(len(X)) if i not in leave_one_out_indices]
    X_train = X[train_indices]
    X_test = X[leave_one_out_indices]
    y_train = y[train_indices]
    y_test = y[leave_one_out_indices]
    return X_train, X_test, y_train, y_test

    if dataset == None:
        info = stress_info.Info(dataset_name = 'mald_all')
        dataset = info.xy(layer = layer, section = 'vowel')

def train_mlp_classifier(X,y, vowel, n = None, leave_one_in = False):
    if leave_one_in: function = make_leave_one_in_train_test_sets
    else: function = make_leave_one_out_train_test_sets
    X_train, X_test, y_train, y_test = function(X,y, vowel, n = n) 
    clf=MLPClassifier(random_state=1,max_iter=300)
    clf.fit(X_train, y_train)
    hyp = clf.predict(X_test)
    return y_test, hyp, clf

def train_classifiers(stress_info = None, layers = layers, 
    overwrite = False, leave_one_in = False):
    '''train mlp classifiers based on the data structure hidden_states.'''
    if not stress_info: stress_info = Info(dataset_name = 'mald_all')
    for layer in layers:
        print('starting on',layer)
        X, y = stress_info.xy(layer = layer, section = 'vowel')
        for vowel in mald_vowels:
            print('training', layer, vowel)
            f = _make_perceptron_filename(layer, vowel, leave_one_in = False)
            if os.path.exists(f) and not overwrite: 
                print('file exists, skipping', f)
                continue
            y_test, hyp, clf = train_mlp_classifier(X,y,vowel, 
                leave_one_in = False)
            save_performance(y_test, hyp, layer, vowel, leave_one_in) 
            with open(f, 'wb') as fout:
                pickle.dump(clf,fout)

def train_all_classifiers(stress_info = None): 
    for leave_one_in in [True, False]:
        if leave_one_in: print('training leave one in classifiers') 
        else: print('training leave one out classifiers')
        train_classifiers(stress_info = stress_info, 
            leave_one_in = leave_one_in)

def _make_dir_and_name(leave_one_in = False):
    if leave_one_in: 
        name = 'leave_one_in'
        directory = locations.leave_one_in_dir
    else: 
        name = 'leave_one_out'
        directory = locations.leave_one_out_dir
    return name, directory

def _make_perceptron_filename(layer, vowel, leave_one_in = False):
    name, directory = _make_dir_and_name(leave_one_in)
    filename = directory + 'perceptron_'+name+'_'+str(layer)+'_'+vowel+'.pickle'
    return filename



def save_performance(gt, hyp, layer, vowel, leave_one_in):
    name, directory = _make_dir_and_name(leave_one_in)
    mcc = round(matthews_corrcoef(gt, hyp), 3)
    cr = classification_report(gt, hyp)
    a = accuracy_score(gt, hyp)
    report = {'classification report': cr, 'mcc': mcc, 'accuracy': a}
    f = directory + 'score_' + name 
    f +=  '_' + str(layer)+'_'+ vowel + '.json'
    with open(f, 'w') as fout:
        json.dump(d, fout)
    return d

from . import timestamps_to_indices as tti
from . import locations
import glob
import json
import numpy as np
import pickle
import random

class Hidden_states:
    def __init__(self, model_name = '', ctc = True, vocab = None):
        self.hidden_states = []
        self.filenames = []
        self.model_name = model_name
        self.ctc = ctc
        self.vocab = vocab

    def add_phrase_hidden_states(self,phrase_hidden_states):
        self.hidden_states.extend(phrase_hidden_states.hidden_states)

    def add_hidden_states(self,hidden_states, filename = None):
        self.hidden_states.extend(hidden_states.hidden_states)
        if filename: self.filenames.append(filename)

    @property
    def char_dict(self):
        if hasattr(self,'_char_dict'): 
            if len(self._char_dict) == len(self.hidden_states):
                return self._char_dict
        self._char_dict = {}
        for x in self.hidden_states:
            if x.char not in self._char_dict.keys():self._char_dict[x.char] = []
            self._char_dict[x.char].append( x )
        return self._char_dict

    @property
    def layer_dict(self):
        if hasattr(self,'_layer_dict'): 
            if len(self._layer_dict) == len(self.hidden_states):
                return self._layer_dict
        self._layer_dict = {}
        for x in self.hidden_states:
            i = x.hidden_state_layer_index
            if i not in self._layer_dict.keys():self._layer_dict[i] = []
            self._layer_dict[i].append( x )
        return self._layer_dict
                
    def count_char_states(self, d = {}):
        layer_index = self.hidden_states[0].hidden_state_layer_index
        for char, hidden_states in self.char_dict.items():
            if char not in d.keys(): d[char] = 0
            for hidden_state in hidden_states:
                if hidden_state.hidden_state_layer_index == layer_index:
                    d[char] += 1
        return d

    def to_dataset(self, layer):
        if not self._check_has_layer(layer): 
            raise ValueError('layer:', layer, 'not in collected hidden states')
        X,y =[], []
        for x in self.layer_dict[layer]:
            X.append(x.vector)            
            y.append(self.vocab[x.char])
        return np.array(X), np.array(y)

    def _check_has_layer(self,layer):
        for x in self.hidden_states[:50]:
            if x.hidden_state_layer_index == layer: return True
        return False
            


class Phrase_hidden_states:
    def __init__(self,outputs,indices, phrase_index, cgn_id, vocab=None,
        hidden_state_layers = [1,6,12,18,14], extract_logit = True):
        self.outputs = outputs
        self.nhidden_state_frames = self.outputs.hidden_states[0][0].shape[0]
        self.indices = indices
        self.phrase_index = phrase_index
        self.cgn_id = cgn_id
        self.vocab = vocab
        if vocab: self.reverse_vocab = reverse_vocab(self.vocab)
        else: 
            self.reverse_vocab = None
            self.extract_logit = False
        self.hidden_state_layers = hidden_state_layers
        self.prelabeled_index_dict = tti.cgn_id_to_index_dict(cgn_id)
        self.extract_logit = extract_logit
        self.make_hidden_states()

    def make_hidden_states(self):
        self.hidden_states = []
        self.char_dict = {}
        self.layer_dict = {}
        for phrase_frame_index, label_index in self.indices:
            if phrase_frame_index >= self.nhidden_state_frames: continue
            d = {'cgn_id':self.cgn_id,'frame_index':label_index}
            d['phrase_frame_index'] = phrase_frame_index
            d['phrase_index'] = self.phrase_index
            if self.extract_logit: 
                d['logit_char'] = self._extract_logit_char(phrase_frame_index)
            if label_index in self.prelabeled_index_dict.keys():
                d['char'] = self.prelabeled_index_dict[label_index]
            else: continue
            if d['char'] == ' ' and random.randint(0,20) != 42: continue
            self._extract_hidden_state_layers(d)

    def _extract_hidden_state_layers(self,d):
        for layer_index in self.hidden_state_layers:
            d['hidden_state_layer_index'] = layer_index
            d['vector'] = self._get_hidden_state(layer_index,
                d['phrase_frame_index'])
            self.hidden_states.append(Hidden_state(**d))
            if d['char'] not in self.char_dict.keys():
                self.char_dict[d['char']] = []
            if layer_index not in self.layer_dict.keys():
                self.layer_dict[layer_index] = []
            self.char_dict[d['char']].append( self.hidden_states[-1] )
            self.layer_dict[layer_index].append( self.hidden_states[-1] )


    def _extract_logit_char(self,phrase_frame_index):
        logit_frame =self.outputs.logits[0][phrase_frame_index].detach().numpy()
        i = np.argmax(logit_frame)
        return self.reverse_vocab[i]

    def _get_hidden_state(self, layer_index, phrase_frame_index):
        layer = self.outputs.hidden_states[layer_index]
        if layer.device.type == 'cuda': layer = layer.cpu()
        return layer[0][phrase_frame_index]
            

class Hidden_state:
    def __init__(self,vector,char,hidden_state_layer_index, logit_char = None, 
        cgn_id = None, frame_index = None,phrase_frame_index = None, 
        phrase_index = None):
        self.vector = np.array(vector)
        self.char = char
        self.hidden_state_layer_index = hidden_state_layer_index
        self.logit_char = logit_char
        self.cgn_id = cgn_id
        self.frame_index = frame_index
        self.phrase_frame_index = phrase_frame_index
        self.phrase_index = phrase_index

    def __repr__(self):
        m = self.char + ' '
        m += self.cgn_id + ' '
        m += str(self.hidden_state_layer_index) 
        if self.logit_char:
            m += ' ' + self.logit_char
        return m
    

def reverse_vocab(vocab):
    return {v:k for k,v in vocab.items()}
         

def collect_hidden_states(location = locations.ctc_hidden_states_dir,
    n_files = 100, vocab = None):
    fn = glob.glob(location + '*.pickle')
    if not vocab: vocab = load_vocab()
    if n_files == 'all' or n_files > len(fn): n_files == len(fn)
    if n_files < len(fn): fn = random.sample(fn, n_files)
    hs = Hidden_states(vocab = vocab)
    for f in fn:
        fin = open(f,'rb')
        x = pickle.load(fin)
        hs.add_hidden_states(x, f)
    return hs


def load_vocab(vocab_filename = None):
    if not vocab_filename:
        fin = open(locations.cache_dir+ 'vocab.json')
    else:
        fin = open(vocab_filename)
    vocab = json.load(fin)
    del vocab['[PAD]']
    del vocab['[UNK]']
    del vocab['|']
    vocab[' '] = 0
    return vocab

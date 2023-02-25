from utils import hidden_state
from utils import perceptron
from utils import phonemes
from utils import to_vectors
from utils import identifiers
from utils import locations
from text.models import Textgrid
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

def load_kl_audio(cgn_id):
    f = locations.kl_audio_dir + cgn_id + '_kl.pickle'
    with open(f,'rb') as fin:
        kla = pickle.load(fin)
    return kla


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
            v = frame_lists_to_kl_vector(v,kl_type)
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
        


def cgn_id_to_kl_audio_filename(cgn_id):
    f = locations.kl_audio_dir + cgn_id + '_kl.pickle'
    return f

def make_kl_audios_for_cgn_component(component, overwrite = False):
    t = Textgrid.objects.filter(component__name = component)
    for x in t:
        filename = cgn_id_to_kl_audio_filename(x.cgn_id)
        print('\n','-'*90,'\nhandling',filename,'\n','-'*90,'\n' )
        if os.path.isfile(filename) and not overwrite: 
            print('already done, skipping')
            continue
        kla = KLAudio(x.cgn_id)
        with open(filename, 'wb') as fout:
            pickle.dump(kla,fout)
        del kla

def get_vocabs():
    '''get the vocabs to get the indices of the phonemes 
    to translate the indices in the prob matrix to phonemes
    '''
    vocab = hidden_state.load_vocab()
    ipa_vocab = phonemes.convert_simple_sampa_vocab_to_ipa_symbols(vocab)
    ipa_reverse_vocab = hidden_state.reverse_vocab(ipa_vocab)
    return vocab, ipa_vocab, ipa_reverse_vocab

vocab, ipa_vocab, ipa_reverse_vocab = get_vocabs()

def matrix_to_phoneme_labels(matrix):
    '''get the winning phoneme label for each column in the prob matrix.'''
    indices = np.argmax(matrix,1)
    phoneme_labels = []
    for index in indices:
        phoneme_labels.append(ipa_reverse_vocab[index])
    return phoneme_labels
        
def synthetic_probability_distribution_to_indices(spd):
    '''find the correct index (in the prob matrix) for each phoneme
    in the synthetic prob distribution (based on the bpc)
    '''
    indices = []
    for phoneme in spd.keys():
        indices.append(ipa_vocab[phoneme])
    return indices

def normalize_prob_vector(v, epsilon = 0.00001):
    '''normalize a prob vector
    the mlp prob output for the phonemes label for a given frame
    need to be normalized after removing the winning (and other phonemes)
    from the mlp output
    v   prob vector from the prob matrix corresponding to a given frame
    epsilon     small value to prevend 0 errors in the kl computation
    '''
    v = v+ epsilon
    return v / np.sum(v)

def compute_mlp_output_and_synthetic_bpc(matrix):
    '''compute the kl divergence for each frame in a prob matrix
    the output from a mlp classifier for a section of audio.
    '''
    bpcs = phonemes.make_bpcs()
    phoneme_labels = matrix_to_phoneme_labels(matrix)
    output = []
    for i,phoneme in enumerate(phoneme_labels):
        try:bpc = bpcs.find_bpc(phoneme)
        except: 
            output.append([phoneme,i,None,None,None])
            continue
        spd = bpc.synthetic_probability_distribution(phoneme)
        indices = synthetic_probability_distribution_to_indices(spd)
        phons = get_phonemes(indices)
        p = normalize_prob_vector(matrix[i,indices])
        q = list(spd.values())
        output.append([phoneme,i,kl_divergence(p,q), p, phons])
    return output

def kl_divergence(p,q):
    '''compute the kullback leibler divergence
    p   observed probs
    q   model probs
    '''
    return np.sum(p*np.log(p/q))

def get_phonemes(indices):
    phons = ''
    for i in indices:
        phons+= ipa_reverse_vocab[i]
    return phons

        
class KLAudio:
    def __init__(self, cgn_id):
        self.cgn_id = cgn_id
        self.component = identifiers.cgn_id_to_component(self.cgn_id).name
        self.compute_klphrases()
        self._set_layer_dicts()

    def __repr__(self):
        m = self.cgn_id + ' '
        m += 'nphrases: ' + str(len(self.klphrases_ctc))
        return m

    def compute_klphrases(self):
        textgrid = Textgrid.objects.get(cgn_id = self.cgn_id)
        self.klphrases_pretrained = []
        self.klphrases_ctc = []
        for index, phrase in enumerate(textgrid.phrases()):
            start_time = phrase[0].start_time
            end_time = phrase[-1].end_time
            x = [textgrid.audio.filename,index,start_time,end_time]
            self.klphrases_pretrained.append( KLPhrase(*x,ctc=False) )
            self.klphrases_ctc.append( KLPhrase(*x,ctc=True) )

    def _add_phrase_layer_dict(self,phrase, d):
        for layer in phrase.layer_names:
            frames = phrase.klframe_layer_dict[layer]
            if not layer in d.keys(): d[layer] = []
            d[layer].extend(frames)

    @property
    def klframe_layer_dict(self):
        if hasattr(self,'_klframe_layer_dict'): 
            return self._klframe_layer_dict
        self._klframe_layer_dict = {'pretrained':{},'ctc':{}}
        for phrase in self.klphrases_pretrained:
            self._add_phrase_layer_dict(
                phrase,
                self.klframe_layer_dict['pretrained'])
        for phrase in self.klphrases_ctc:
            self._add_phrase_layer_dict(
                phrase,
                self.klframe_layer_dict['ctc'])
        return self._klframe_layer_dict

    @property
    def klframes_pretrained(self):
        return self._get_frames('_pretrained')

    @property
    def klframes_ctc(self):
        return self._get_frames('_ctc')

    @property
    def kl_pretrained_layers(self):
        return self.klphrases_pretrained[0].layer_names

    @property
    def kl_ctc_layers(self):
        return self.klphrases_ctc[0].layer_names

    @property
    def n_total_frames(self):
        nframes = 0
        for phrase in self.klphrases_pretrained:
            nframes += phrase.n_total_frames
        return nframes

    def _make_n_usabel_frames(self, model_type, d):
        if model_type == 'pretrained': phrases = self.klphrases_pretrained
        else: phrases = self.klphrases_ctc
        for phrase in phrases:
            for layer, n in phrase.n_usable_frames.items():
                if layer not in d[model_type].keys(): 
                    d[model_type][layer] = n
                else:
                    d[model_type][layer] += n

    @property
    def n_usable_frames(self):
        d = {'pretrained':{},'ctc':{}}
        for key in d.keys():
            self._make_n_usabel_frames(key, d)
        return d

    def _make_layer_vector_dict(self, model_type, d):
        if model_type == 'pretrained': phrases = self.klphrases_pretrained
        else: phrases = self.klphrases_ctc
        phrases_to_layer_vector_dict(phrases,model_type,d)

    @property
    def layer_vector_dict(self):
        d = {'pretrained':{},'ctc':{}}
        for key in d.keys():
            self._make_layer_vector_dict(key, d)
        return d

    def _get_frames(self,frame_type):
        if hasattr(self,'_klframes_' + frame_type):
            return getattr(self,'_klframes_' + frame_type)
        if 'pretrained' in frame_type:phrases = self.klphrases_pretrained
        else: phrases = self.klphrases_ctc
        output = []
        for x in phrases:
            for frame in x.klframes:
                if not frame.kl: continue
                output.append(frame)
        setattr(self,'_klframes_'+frame_type, output)
        return getattr(self,'_klframes_' + frame_type)

class KLPhrase:
    def __init__(self,audio_filename, index, start_time, end_time, ctc):
        self.audio_filename = audio_filename
        self.phrase_index = index
        self.start_time = start_time
        self.end_time = end_time
        self.ctc = ctc
        self.compute_klframes()

    def __repr__(self):
        return 'KLPhrase: ' + self.audio_filename

    @property
    def n_total_frames(self):
        return len(self.klframe_layer_dict[1])

    @property
    def n_usable_frames(self):
        d = {}
        for layer in self.layer_names:
            frames = self.klframe_layer_dict[layer]
            d[layer] = len([x for x in frames if x.kl != None])
        return d

    @property
    def layer_names(self):
        k = list(self.klframe_layer_dict.keys())
        if 'cnn_features' in k:
            x = k.pop(k.index('cnn_features'))
            k = [x] + k
        return k

    @property
    def layer_vector_dict(self):
        d = {}
        for layer in self.layer_names:
            frames = self.klframe_layer_dict[layer]
            d[layer] = np.array([x.kl for x in frames if x.kl != None])
        return d
            

    def compute_klframes(self):
        self.klframe_layer_dict = {}
        self.klframes = []
        hs = to_vectors.audio_to_hidden_states(
            audio_filename = self.audio_filename,
            start = self.start_time,
            end = self.end_time,
            ctc = self.ctc)
        for layer in hs.layer_dict.keys():
            print('handling layer:',layer)
            self.handle_layer(hs, layer)

    def handle_layer(self, hs, layer):
        self.klframe_layer_dict[layer] = []
        clf = perceptron.load_perceptron(layer = layer, ctc = self.ctc)
        x = hs.to_dataset(layer)[0]
        matrix = clf.predict_proba(x)
        output = compute_mlp_output_and_synthetic_bpc(matrix)
        for line in output:
            phoneme, i, kl, probs, phons = line
            klf = KLFrame(phoneme, i, kl, probs, phons,layer, self.ctc)
            self.klframe_layer_dict[layer].append(klf)
            self.klframes.append(klf)
        
    
class KLFrame:
    def __init__(self, phoneme, i, kl,probs,phons, layer, ctc):
        self.phoneme = phoneme
        self.frame_index = i
        self.kl = kl
        self.probability_vector = probs
        self.phonemes_vector = phons
        self.layer = layer
        self.ctc = ctc

    def __repr__(self):
        kl = str(round(self.kl,2)) if self.kl != None else 'NA'
        model_type = 'ctc' if self.ctc else 'pretrained'
        m = self.phoneme + ' ' + kl + ' ' 
        m += str(self.layer) + ' ' + model_type
        return m

    @property
    def phoneme_probability_vector(self):
        return sorted_phoneme_probs(self.phonemes_vector,
            self.probability_vector)

    @property
    def bpc(self):
        bpcs = phonemes.make_bpcs()
        try: bpc = bpcs.find_bpc(self.phoneme)
        except ValueError: return None
        return bpc
        

    
def sorted_phoneme_probs(phonemes_vector,probability_vector):
    output = []
    for phon, prob in zip(phonemes_vector,probability_vector):
        output.append([phon,prob])
    output = sorted(output, key = lambda x: x[1], reverse = True)
    return output
        

def cgn_ids_to_layer_vector_dict(cgn_ids):
    d = {'pretrained':{},'ctc':{}}
    for cgn_id in cgn_ids:
        kla = load_kl_audio(cgn_id)
        phrases_to_layer_vector_dict(kla.klphrases_ctc,'ctc',d)
        phrases_to_layer_vector_dict(
            kla.klphrases_pretrained,
            'pretrained',
            d)
    return d

def phrases_to_layer_vector_dict(klphrases,model_type, d):
    for phrase in klphrases:
        for layer, vector in phrase.layer_vector_dict.items():
            if layer not in d[model_type].keys(): 
                d[model_type][layer] = vector
            else:
                old_vector = d[model_type][layer] 
                d[model_type][layer]=np.concatenate([old_vector,vector])
    return d

def _add_layer_frame_dict(d,klad):
    for layer, frames in klad.items():
        if not layer in d.keys():
            d[layer] = []
        d[layer].extend(frames)

def cgn_ids_layer_frame_dict(cgn_ids):
    '''creates a layers frame dict based on all frames of the audio
    files corresponding to the cgn identifiers.
    '''
    d = {'pretrained':{},'ctc':{}}
    for cgn_id in cgn_ids:
        kla = load_kl_audio(cgn_id)
        for mt in ['pretrained','ctc']:
            _add_layer_frame_dict(d[mt],kla.klframe_layer_dict[mt])
    return d

def _add_bpc_frames(o, d, bpc):
    '''add only frames belong to a specific bpc.'''
    for layer,frames in d.items():
        if layer not in o.keys(): o[layer] = []
        selected = [x for x in frames if bpc.part_of(x.phoneme)]
        if selected:
            o[layer].extend(selected)
        
        
def cgn_ids_bpc_frame_dict(cgn_ids = None, d = None):
    '''create a dictionary of bpcs with frame dictionaries.'''
    if not d: d = cgn_ids_layer_frame_dict(cgn_ids)
    o = {}
    bpcs = phonemes.make_bpcs()
    for name in bpcs.names:
        o[name] = {'pretrained':{},'ctc':{}}
        for mt in ['pretrained','ctc']:
            _add_bpc_frames(o[name][mt],d[mt], bpcs.bpcs[name])
    return o

    
def frame_lists_to_kl_vector(frame_lists, kl_type = 'normal'):
    '''maps klframes to a vector of kl values.'''
    o = []
    for frame_list in frame_lists:
        if kl_type == 'normal':
            o.append([x.kl for x in frame_list if x.kl])
        elif kl_type == 'random':
            o.append(_frame_list_to_random_kl_vector(frame_list))
        elif kl_type == 'diff':
            o.append(_frame_list_to_diff_kl_vector(frame_list))
    return o

def _frame_list_to_random_kl_vector(frame_list):
    '''maps klframes to a vector of kl values based on a random
    model distribution.'''
    has_random = False
    for x in frame_list:
        if x.kl and hasattr(x,'kl_random'): has_random = True
    if has_random:
        return [x.kl_random for x in frame_list if x.kl]
    else:
        return [frame_to_random_kl(x) for x in frame_list if x.kl]

def _frame_list_to_diff_kl_vector(frame_list):
    '''maps klframes to a vector of kl values based on the 
    difference between random and model distribution.'''
    has_random = False
    for x in frame_list:
        if x.kl and hasattr(x,'kl_random'): has_random = True
    if not has_random:
        _ = [frame_to_random_kl(x) for x in frame_list if x.kl]
    return [x.kl_diff for x in frame_list if x.kl]

def frame_to_random_kl(frame):
    '''compute kl based on a random probability distribution
    '''
    rpd = frame.bpc.random_other_probability_distribution(frame.phoneme)
    p_vector = []
    random_q_vector = []
    for phoneme, prob in frame.phoneme_probability_vector:
        p_vector.append(prob)
        random_q_vector.append(rpd[phoneme])
    rkl = kl_divergence(np.array(p_vector),np.array(random_q_vector))
    frame.kl_random = rkl
    frame.kl_diff = frame.kl - frame.kl_random
    return rkl
    
    

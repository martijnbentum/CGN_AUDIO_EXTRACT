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


def boxplot_kl_frames(layer_to_vector_dict, name = '', filename = ''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    axis = [ax1,ax2]
    for i,key in enumerate(layer_to_vector_dict.keys()):
        v = list(layer_to_vector_dict[key].values())
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
        
def synthetic_probability_ditribution_to_indices(spd):
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
    '''compute the kl divergence for each fram in a prob matrix
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
        spd = bpc.synthetic_probability_ditribution(phoneme)
        indices = synthetic_probability_ditribution_to_indices(spd)
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

    
def sorted_phoneme_probs(phonemes_vector,probability_vector):
    output = []
    for phon, prob in zip(phonemes_vector,probability_vector):
        output.append([phon,prob])
    output = sorted(output, key = lambda x: x[1], reverse = True)
    return output
        

def cgn_ids_to_layer_vector_dict(cgn_ids, model_type =None):
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

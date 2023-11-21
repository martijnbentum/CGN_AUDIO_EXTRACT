'''module for working with codevectors
a codevector is a concatenation of two codebook vectors
a codevector is represented by a pair of indices
'''

from . import awd
import copy
import glob
import json
from matplotlib import pyplot as plt
import numpy as np
import pickle
from utils import locations
from utils import general
from utils import phonemes

fn = glob.glob(locations.codebook_indices_dir + '*.npy')


def load_codebook_indices(filename):
    '''load codebook indices from a numpy file
    the indices are based on a cgn audio file
    '''
    return np.load(filename)

def load_codebook(filename=locations.codebook_indices_dir + 'codebook.npy'):
    '''load the codebook
    the correct codebook depends on the module used to compute the 
    codebook indices
    '''
    return np.load(filename)

class Frames:
    '''object to handle wav2vec2 linking frames to phoneme transcriptions
    each frame has codebook indices representing a codevector
    each frame can be linked to a phoneme via the awd transcription
    each frame has a start and end time
    the frames have a step of 20ms and a duration of 25ms
    '''
    def __init__(self, fn = fn, awds = None):
        if not awds:
            self.awds = pickle.load(open('../awds.pickle', 'rb'))
        else:self.awds = awds
        self.frames = []
        self.filenames = []
        for f in fn:
            self.add_file(f)
        self.frame_set = set(self.frames)

    def add_file(self, filename):
        if 'codebook' in filename: return
        if filename in self.filenames:
            print('already added', filename)
            return
        self.filenames.append(filename)
        codebook_indices = load_codebook_indices(filename)
        self._add_frames(codebook_indices, filename)

    def __repr__(self):
        m = 'nframes: '+str(len(self.frames)) 
        m += ' ' + str(len(set(self.frames)))
        return m

    def _add_frames(self, codebook_indices, filename):
        for index, ci in enumerate(codebook_indices):
            frame = Frame(index, ci, self, filename)
            self.frames.append(frame)

    def get_codevector(self, codebook_indices):
        i1, i2 = codebook_indices
        half = self.codebook.shape[1] // 2
        q1 = self.codebook[i1]
        q2 = self.codebook[i2]
        return np.concatenate([q1, q2], axis=0)

    @property
    def codebook(self):
        return load_codebook()

class Frame:
    '''object to store information for a specific frame
    '''
    def __init__(self, index, codebook_indices, parent, filename):
        self.index = index
        self.codebook_indices = codebook_indices
        self.parent = parent
        self.filename = filename
        self.cgn_id = filename.split('_')[-1].split('.')[0]
        self.awd_filename = locations.local_awd + self.cgn_id + '.awd'
        self.start, self.end = frame_index_to_times(index)
        self.i1, self.i2 = map(int,self.codebook_indices)
        self.key = (self.i1, self.i2)

    def __repr__(self):
        m = str(self.index) + ' ' + str(self.codebook_indices)
        m += ' ' + str(self.start) + ' ' + str(self.end)
        return m

    def __eq__(self, other):
        if type(self) != type(other):return False
        return self.i1== other.i1 and self.i2 == other.i2

    def __hash__(self):
        return hash((self.i1, self.i2))

    @property
    def awd(self):
        '''the forced aligned transcription file from CGN
        to link a frame to a specific phoneme
        '''
        if hasattr(self,'_awd'): return self._awd
        self._awd = self.parent.awds.get_awd(self.awd_filename)
        return self._awd

    @property
    def phoneme(self):
        '''the phoneme linked to the frame
        the selected phoneme is the one with the largest overlap
        with the frame start and end time
        '''
        if hasattr(self,'_phoneme') and self._phoneme: 
            return self._phoneme
        s1, e1 = self.start,self.end
        self._phoneme = None
        self._phoneme_duration = 0
        for phoneme in self.awd.phonemes:
            s2, e2 = phoneme.start, phoneme.end
            if general.overlap(s1,e1,s2,e2):
                duration = general.overlap_duration(s1,e1,s2,e2)
                if self._phoneme and duration < self._phoneme_duration: 
                    continue
                self._phoneme = phoneme
                self._phoneme_duration = duration
        return self._phoneme

    @property
    def codevector(self):
        '''the codevector for the frame
        based on the indices
        '''
        return self.parent.get_codevector(self.codebook_indices)

            


def frame_index_to_times(index, step = 0.02, duration = 0.025):
    '''compute the start and end time for a frame
    '''
    start = index * step
    end = start + duration
    return round(start,3), round(end,3)
        
    
    
def make_frame_dict(frames):
    '''make a dictionary of frames
    the identifier for a frame is the pair of indices linking it to the 
    codebook
    '''
    d = {}
    for frame in frames:
        key = (frame.i1, frame.i2)
        if not key in d.keys(): d[key] = []
        d[key].append(frame)
    return d

def frames_to_phoneme_counter(frames):
    '''each frame token is linked to a phoneme
    a frame type is linked to multiple phonemes
    count the phonemes that a frame type is linked to
    '''
    d = {}
    for frame in frames:
        phoneme = frame.phoneme
        if phoneme.label == '': phoneme.label = 'silence'
        if not phoneme.label in d.keys(): d[phoneme.label] = 0
        d[phoneme.label] += 1
    return d


def dict_to_sorted_dict(d):
    '''sort a dict based on the values
    '''
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

def count_dict_to_probability_dict(d):
    '''convert a count dict to a probability dict
    '''
    od = copy.copy(d)
    total = sum(d.values())
    for key in d.keys():
        od[key] /= total
    return od

def frames_to_count_dict(frames, save = False):
    '''for each codevector, count the phonemes that are linked to it
    '''
    key = frames[0].key
    d = frames_to_phoneme_counter(frames)
    d = dict_to_sorted_dict(d)
    if save:
        filename = locations.codebook_indices_phone_counts_dir 
        filename += '-'.join(map(str,key)) + '.json'
        json.dump(d, open(filename, 'w'))
    return d

    
def load_count_dict(filename):
    '''load a codevector phoneme count dict.'''
    d = json.load(open(filename))
    od ={}
    for key in d.keys():
        if key[0] == '!': continue
        if key == '[]': continue
        od[key] = d[key]
    return od


def load_all_count_dicts():
    '''load all codevector phoneme count dicts.'''
    path = locations.codebook_indices_phone_counts_dir + '*.json'
    filenames = glob.glob(path)
    to_name = codevector_json_filename_to_name
    return dict([[to_name(f), load_count_dict(f)] for f in filenames])

def _get_all_phonemes(count_dicts):
    '''list all phonemes present in a set of codevector count dicts.'''
    d = count_dicts
    p = []
    for v in d.values():
        for k in v.keys():
            if k not in p: p.append(k)
    return p

def create_phoneme_codevector_counts(p, d):
    '''create dictionary that maps a phoneme to a codevector count dict.'''
    output_dict = {}
    for phoneme in p:
        output_dict[phoneme] = {}
        for codevector_filename, phoneme_count_dict in d.items():
            name = codevector_json_filename_to_name(codevector_filename)
            if phoneme not in phoneme_count_dict.keys(): phoneme_count = 0
            else: phoneme_count = phoneme_count_dict[phoneme]
            output_dict[phoneme][name] = phoneme_count
    return output_dict
        

def codevector_json_filename_to_name(filename):
    '''map the codevector phoneme count dict json filename to 
    the codevector name (the codebook indices: index1-index2)
    '''
    name = filename.split('/')[-1].split('.')[0]
    return name
    
'''
OBSOLETE, creates the same matrix as the create_matrix_phoneme_counts function
def create_matrix_codevector_counts(p, d):
    rows, columns = len(p), len(d['silence'])
    m = np.zeros((rows, columns))
    for i, phoneme in enumerate(p):
        codevector_counts = d[phoneme]
        cc = codevector_counts
        for j, phoneme_prob in enumerate(sorted(cc.values())):
            m[i,j] = phoneme_prob
    return m
'''

def create_matrix_phoneme_counts(p, d):
    '''create a matrix that with a phoneme per row and codevectors
    per column. The matrix values are the counts of the phoneme each 
    codevector column.
    '''
    rows, columns = len(p), len(d)
    m = np.zeros((rows, columns))
    for i, phoneme in enumerate(p):
        for j, codevector_phoneme_counts in enumerate(d.values()):
            cpc = codevector_phoneme_counts
            if phoneme not in cpc.keys():
                m[i,j] = 0
                continue
            m[i,j] = cpc[phoneme]
    return m
            

def sort_probability_dict(d):
    '''sort a probability dict based on the values.
    '''
    o = (sorted(d.items(), key=lambda item: list(item[1].keys())[0], 
        reverse=True))
    return dict(o)


def compute_phoneme_pdf(d):
    '''computes a probability distribution over phonemes.
    '''
    output_d = {}
    p = _get_all_phonemes(d)
    m = create_matrix_phoneme_counts(p, d)
    all_count = np.sum(m)
    for i,phoneme in enumerate(p):
        output_d[phoneme] = np.sum(m[i]) / all_count
    return output_d

def compute_codevector_pdf(d):
    '''computes a probability distribution over codevectors.
    '''
    output_d = {}
    p = _get_all_phonemes(d)
    m = create_matrix_phoneme_counts(p, d)
    all_count = np.sum(m)
    for i,key in enumerate(d.keys()):
        # name = codevector_json_filename_to_name(key)
        output_d[key] = np.sum(m[:,i]) / all_count
    return output_d

def compute_conditional_probability_matrix(d):
    '''plot the conditional probability matrix for P(phoneme | codevector).
    '''
    p = _get_all_phonemes(d)
    cpdf = compute_codevector_pdf(d)
    m = create_matrix_phoneme_counts(p, d)
    all_count = np.sum(m)
    cp_m = np.zeros(m.shape)
    for i, phoneme in enumerate(p):
        for j, codevector_name in enumerate(d.keys()):
            cp_m[i,j] = m[i,j] / all_count / cpdf[codevector_name]
    return cp_m

def plot_conditional_probability_matrix(d):
    '''plot the conditional probability matrix for P(phoneme | codevector).
    '''
    p = _get_all_phonemes(d)
    m = compute_conditional_probability_matrix(d)
    row_index_max_value = np.argmax(m, axis=0)
    column_indices = np.argsort(row_index_max_value)
    m = m[:,column_indices]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.matshow(m, aspect = 150, cmap = 'binary')
    ax.yaxis.set_ticks(range(len(p)),p)
    plt.show()
        
   


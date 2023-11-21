from . import awd
import copy
import glob
import json
import numpy as np
import pickle
from utils import locations
from utils import general
from utils import phonemes

fn = glob.glob(locations.codebook_indices_dir + '*.npy')


def load_codebook_indices(filename):
    return np.load(filename)

def load_codebook(filename=locations.codebook_indices_dir + 'codebook.npy'):
    return np.load(filename)

class Frames:
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
        if hasattr(self,'_awd'): return self._awd
        self._awd = self.parent.awds.get_awd(self.awd_filename)
        return self._awd

    @property
    def phoneme(self):
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
        return self.parent.get_codevector(self.codebook_indices)

            


def frame_index_to_times(index, step = 0.02, duration = 0.025):
    start = index * step
    end = start + duration
    return round(start,3), round(end,3)
        
    
    
def make_frame_dict(frames):
    d = {}
    for frame in frames:
        key = (frame.i1, frame.i2)
        if not key in d.keys(): d[key] = []
        d[key].append(frame)
    return d

def frames_to_phoneme_counter(frames):
    d = {}
    for frame in frames:
        phoneme = frame.phoneme
        if phoneme.label == '': phoneme.label = 'silence'
        if not phoneme.label in d.keys(): d[phoneme.label] = 0
        d[phoneme.label] += 1
    return d


def dict_to_sorted_dict(d):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

def count_dict_to_probability_dict(d):
    od = copy.copy(d)
    total = sum(d.values())
    for key in d.keys():
        od[key] /= total
    return od

def frames_to_count_dict(frames, save = False):
    key = frames[0].key
    d = frames_to_phoneme_counter(frames)
    d = dict_to_sorted_dict(d)
    if save:
        filename = locations.codebook_indices_phone_counts_dir 
        filename += '-'.join(map(str,key)) + '.json'
        json.dump(d, open(filename, 'w'))
    return d

    
def load_count_dict(filename):
    d = json.load(open(filename))
    od ={}
    for key in d.keys():
        if key[0] == '!': continue
        if key == '[]': continue
        od[key] = d[key]
    return od


def load_all_count_dicts():
    path = locations.codebook_indices_phone_counts_dir + '*.json'
    filenames = glob.glob(path)
    return dict([[f, load_count_dict(f)] for f in filenames])

def _get_all_phonemes(count_dicts):
    d = count_dicts
    p = []
    for v in d.values():
        for k in v.keys():
            if k not in p: p.append(k)
    return p

def create_phoneme_codevector_counts(p, d):
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
    name = filename.split('/')[-1].split('.')[0]
    return name
    
def create_matrix_codevector_counts(p, d):
    rows, columns = len(p), len(d['silence'])
    m = np.zeros((rows, columns))
    for i, phoneme in enumerate(p):
        codevector_counts = d[phoneme]
        ccp = codevector_counts
        # ccp = count_dict_to_probability_dict(codevector_counts)
        for j, phoneme_prob in enumerate(sorted(ccp.values())):
            m[i,j] = phoneme_prob
    return m

def create_matrix_phoneme_counts(p, d):
    rows, columns = len(p), len(d)
    m = np.zeros((rows, columns))
    for i, phoneme in enumerate(p):
        for j, codevector_phoneme_counts in enumerate(d.values()):
            # cpp = count_dict_to_probability_dict(codevector_phoneme_counts)
            cpp = codevector_phoneme_counts
            if phoneme not in cpp.keys():
                m[i,j] = 0
                continue
            m[i,j] = cpp[phoneme]
    return m
            

def plot_codevectors():
    d = load_all_count_dicts()
    p = _get_all_phonemes(d)
    plt.ion()
    fig, ax = plt.subplots(8,8)
    ax.matshow(m, aspect = 150)

def sort_probability_dict(d):
    o = (sorted(d.items(), key=lambda item: list(item[1].keys())[0], 
        reverse=True))
    return dict(o)


def find_indices_of_high_codevectors(phoneme,pcd):
    pass
    '''
    indices = []
    for i, codevector_counts in enumerate(pcd[phoneme]):
        ccp = count_dict_to_probability_dict(codevector_counts)
        
            indices.append(i)
    return indices
    '''

def compute_phoneme_pdf(d):
    output_d = {}
    p = _get_all_phonemes(d)
    m = create_matrix_phoneme_counts(p, d)
    all_count = np.sum(m)
    for i,phoneme in enumerate(p):
        output_d[phoneme] = np.sum(m[i]) / all_count
    return output_d

def compute_codevector_pdf(d):
    output_d = {}
    p = _get_all_phonemes(d)
    m = create_matrix_codevector_counts(p, d)
    all_count = np.sum(m)
    for i,phoneme in enumerate(p):
        output_d[phoneme] = np.sum(m[i]) / all_count
    return output_d


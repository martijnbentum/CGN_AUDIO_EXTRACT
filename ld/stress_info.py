from utils import locations
from utils import audio
import numpy as np
import pickle
import random
from ld import time_index

float_columns='start_time,end_time,vowel_start_time,vowel_end_time'.split(',')

class Info:
    def __init__(self, dataset_name='mald', model_type='wav2vec'):
        if dataset_name == 'mald':
            self.filename = locations.mald_variable_stress_info
        else: raise ValueError('dataset_name must be mald')
        if model_type == 'wav2vec':
            self.variable_stress_wav_dir = locations.mald_variable_stress_wav 
            path = locations.mald_variable_stress_pretrain_vectors 
            self.variable_stress_pretrain_vectors_dir = path
        else: raise ValueError('model_type must be wav2vec')
        self._set_info()

    def _set_info(self):
        self.info = {}
        with open(self.filename) as f:
            self._text = f.read()
        temp = [x.split('\t') for x in self._text.split('\n')]
        self.header = temp[0]
        self.data = temp[1:]
        self.syllables = [Syllable(x, self.header,self) for x in self.data]

    def xy(self, layer='cnn', section = 'syllable', random_gt = False):
        attr_name = '_xy_' + section + '_' + str(layer)
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        X = np.array([x.X(layer, section) for x in self.syllables])
        y = np.array([x.y(random_gt) for x in self.syllables])
        setattr(self, attr_name, (X,y))
        return getattr(self, attr_name)
        


class Syllable:
    def __init__(self, line, header, info):
        self.line = line
        self.header = header
        self.info = info
        self._set_info()

    def _set_info(self):
        for name, value in zip(self.header, self.line):
            if name in float_columns: value = float(value)
            if name == 'stressed': value = value == 'True'
            setattr(self,name,value)
        self.name = self.word_audio_filename.split('.')[0]

    @property
    def wav_filename(self):
        return self.info.variable_stress_wav_dir + self.word_audio_filename

    @property
    def pretrain_vectors_filename(self):
        f = self.info.variable_stress_pretrain_vectors_dir 
        f += self.name + '.pickle'
        return f

    @property
    def pretrain_vectors(self):
        if hasattr(self, '_pretrain_vectors'):
            return self._pretrain_vectors
        with open(self.pretrain_vectors_filename, 'rb') as f:
            self._pretrain_vectors = pickle.load(f)
        return self._pretrain_vectors

    @property
    def sox_info(self):
        if hasattr(self, '_sox_info'):
            return self._sox_info
        temp = audio.sox_info(self.wav_filename)
        self._sox_info = audio.soxinfo_to_dict(temp)
        return self._sox_info

    @property
    def word_duration(self):
        return self.sox_info['duration']
            
    @property
    def start_end_index(self):
        return time_index.time_slice_to_index_slice(
            self.start_time, self.end_time)

    @property
    def start_end_index_time(self):
        return time_index.index_slice_to_time_slice(
            *self.start_end_index)

    @property
    def start_end_time(self):
        return self.start_time, self.end_time

    @property
    def vowel_start_end_index(self):
        return time_index.time_slice_to_index_slice(
            self.vowel_start_time, self.vowel_end_time)

    @property
    def vowel_start_end_index_time(self):
        return time_index.index_slice_to_time_slice(
            *self.vowel_start_end_index)

    @property
    def vowel_start_end_time(self):
        return self.vowel_start_time, self.vowel_end_time

    def _get_feature_vectors(self, layer = 'cnn'):
        if layer == 'cnn':
            return self.pretrain_vectors.extract_features[0].numpy()
        if type(layer) != int:
            raise ValueError('layer must be layer index or "cnn"')
        return self.pretrain_vectors.hidden_states[layer][0].numpy()

    def feature_vectors(self, layer = 'cnn', section = 'syllable'):
        feature_vectors = self._get_feature_vectors(layer)
        if section == 'syllable':
            start_index, end_index = self.start_end_index
        elif section == 'vowel':
            start_index, end_index = self.vowel_start_end_index
        elif section == 'word':
            start_index, end_index = 0, feature_vectors.shape[0]
        return feature_vectors[start_index:end_index]

    def mean_feature_vector(self, layer = 'cnn', section = 'syllable'):
        attr_name = '_' + section + '_mean_feature_vector_' + str(layer)
        if hasattr(self,attr_name):
            return getattr(self,attr_name)
        temp = np.mean(self.feature_vectors(layer, section), axis=0)
        setattr(self, attr_name, temp)
        return getattr(self,attr_name)
    
    def X(self, layer = 'cnn', section = 'syllable'):
        return self.mean_feature_vector(layer, section)

    def y(self, random_gt = False):
        if random_gt: return random.randint(0,1)
        return int(self.stressed)

        


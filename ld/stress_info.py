from utils import locations
from utils import audio
import numpy as np
import pickle
from progressbar import progressbar
import random
from ld import sox
from ld import time_index

float_columns='start_time,end_time,vowel_start_time,vowel_end_time'.split(',')

class Info:
    def __init__(self, dataset_name='mald_variable', model_type='wav2vec'):
        self.dataset_name = dataset_name
        self.model_type = model_type
        if model_type != 'wav2vec': 
            raise ValueError('model_type must be wav2vec')
        if dataset_name == 'mald_variable': self._set_mald_variable()
        elif dataset_name == 'mald_all': self._set_mald_all()
        else: raise ValueError('dataset_name must be mald_variable or mald_all')
        self._set_info()

    def _set_mald_variable(self):
        self.filename = locations.mald_variable_stress_info
        self.wav_dir = locations.mald_variable_stress_wav 
        path = locations.mald_variable_stress_pretrain_vectors 
        self.pretrain_vectors_dir = path
        path = locations.mald_variable_stress_occlusions_pretrain_vectors
        self.occlusions_pretrain_vectors_dir = path

    def _set_mald_all(self):
        self.filename = locations.mald_all_stress_info
        self.wav_dir = locations.mald_word_recordings
        self.pretrain_vectors_dir = locations.mald_pretrain_vectors

    def _set_info(self):
        self.info = {}
        with open(self.filename) as f:
            self._text = f.read()
        temp = [x.split('\t') for x in self._text.split('\n')]
        self.header = temp[0]
        self.data = temp[1:]
        self.syllables = [Syllable(x, self.header,self) for x in self.data]

    def xy(self, layer='cnn', section = 'syllable', random_gt = False,
        occlusion_type = None, n = None):
        attr_name = '_xy_' + section + '_' + str(layer) 
        attr_name += '_' + str(occlusion_type)
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        ot = occlusion_type
        x_values,y_values = [],[]
        if not n: syllables = self.syllables
        else: syllables = self.syllables[:n]
        for syllable in progressbar(syllables):
            x = syllable.X(layer, section, ot)
            if x is None: continue
            x_values.append(x)
            y_values.append(syllable.y(random_gt))
        X = np.array(x_values)
        y = np.array(y_values)
        '''
        X = np.array([x.X(layer, section, ot) for x in self.syllables])
        y = np.array([x.y(random_gt) for x in self.syllables])
        '''
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
        return self.info.wav_dir + self.word_audio_filename

    @property
    def pretrain_vectors_filename(self):
        f = self.info.pretrain_vectors_dir 
        f += self.name + '.pickle'
        return f

    @property
    def occlusions_pretrain_vectors_filename_vowel(self):
        f = self.info.occlusions_pretrain_vectors_dir 
        f += self.name + '_only_vowel.pickle'
        return f

    @property
    def occlusions_pretrain_vectors_filename_syllable(self):
        f = self.info.occlusions_pretrain_vectors_dir 
        f += self.name + '_only_syllable.pickle'
        return f

    def pretrain_vectors(self, occlusion_type = None):
        if not occlusion_type:attr_name = '_pretrain_vectors'
        else: attr_name = '_pretrain_vectors_' + str(occlusion_type)
        if hasattr(self, attr_name):
            return getattr(self,attr_name)
        if occlusion_type: 
            name = 'occlusions_pretrain_vectors_filename_'
            filename = getattr(self,name + occlusion_type)
        else: filename = self.pretrain_vectors_filename

        with open(filename, 'rb') as f:
            vectors = pickle.load(f)
        setattr(self, attr_name, vectors)
        return getattr(self,attr_name)

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

    def _get_feature_vectors(self, layer = 'cnn', occlusion_type = None):
        pretrain_vectors = self.pretrain_vectors(occlusion_type)
        if layer == 'cnn':
            return pretrain_vectors.extract_features[0].numpy()
        if type(layer) != int:
            raise ValueError('layer must be layer index or "cnn"')
        return pretrain_vectors.hidden_states[layer][0].numpy()

    def feature_vectors(self, layer = 'cnn', section = 'syllable',
        occlusion_type = None):
        feature_vectors = self._get_feature_vectors(layer, occlusion_type)
        if section == 'syllable':
            start_index, end_index = self.start_end_index
        elif section == 'vowel':
            start_index, end_index = self.vowel_start_end_index
        elif section == 'word':
            start_index, end_index = 0, feature_vectors.shape[0]
        return feature_vectors[start_index:end_index]

    def mean_feature_vector(self, layer = 'cnn', section = 'syllable',
        occlusion_type = None):
        attr_name = '_' + section + '_mean_feature_vector_' + str(layer)
        attr_name += '_' + str(occlusion_type)
        if hasattr(self,attr_name):
            return getattr(self,attr_name)
        feature_vectors = self.feature_vectors(layer, section, occlusion_type)
        if feature_vectors.size != 0: 
            temp = np.mean(feature_vectors, axis=0)
        else: temp = None
        setattr(self, attr_name, temp)
        return getattr(self,attr_name)
    
    def X(self, layer = 'cnn', section = 'syllable', occlusion_type = None):
        return self.mean_feature_vector(layer, section, occlusion_type)

    def y(self, random_gt = False):
        if random_gt: return random.randint(0,1)
        return int(self.stressed)

        
def occlude_except_syllable(syllable):
    s = syllable
    name = s.word_audio_filename
    input_filename = locations.mald_variable_stress_wav + name
    output_dir = locations.mald_variable_stress_occlusions_wav
    output_filename = output_dir + name.replace('.wav', '_only_syllable.wav')
    print(input_filename,output_filename,s.start_time,s.end_time)
    sox.occlude_other(input_filename, output_filename, s.start_time, s.end_time)


def occlude_except_vowel(syllable):
    s = syllable
    name = s.word_audio_filename
    start_time, end_time = s.vowel_start_time, s.vowel_end_time
    input_filename = locations.mald_variable_stress_wav + name
    output_dir = locations.mald_variable_stress_occlusions_wav
    output_filename = output_dir + name.replace('.wav', '_only_vowel.wav')
    print(input_filename,output_filename,start_time,end_time)
    sox.occlude_other(input_filename, output_filename, start_time, end_time)


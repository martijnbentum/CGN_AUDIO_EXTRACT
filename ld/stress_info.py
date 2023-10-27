from utils import locations
from utils import audio
import pickle

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


class Syllable:
    def __init__(self, line, header, info):
        self.line = line
        self.header = header
        self.info = info
        self._set_info()

    def _set_info(self):
        for name, value in zip(self.header, self.line):
            if name in float_columns: value = float(value)
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
            
        


import glob
import pickle
from praatio import textgrid
from utils import locations

fn = glob.glob(locations.local_awd + '*.awd')

def load_textgrid(filename):
    tg = textgrid.openTextgrid(filename, True)
    tg.filename = filename
    return tg

def get_segment_tier(textgrid):
    for name in textgrid.tierNames:
        if 'SEG' in name:
            return textgrid.getTier(name)
    raise ValueError('No segment tier found',textgrid.tierNames)

def get_word_tier(textgrid):
    for name in textgrid.tierNames:
        if 'FON' in name:
            return textgrid.getTier(name)
    raise ValueError('No fon tier found',textgrid.tierNames)

class Awds:
    def __init__(self, filenames = fn, save = False):
        self.filenames = filenames
        self.awds = []
        self._add_awds()
        if save: self.save()

    def _add_awds(self):
        for filename in self.filenames:
            awd = Awd(filename)
            self.awds.append(awd)

    def get_awd(self, filename):
        for awd in self.awds:
            if awd.filename == filename:
                return awd
        raise ValueError('No awd with filename', filename)

    def save(self):
        f = '../awds.pickle'
        pickle.dump(self, open(f, 'wb'))

class Awd:
    def __init__(self, filename):
        self.filename = filename
        self.textgrid = load_textgrid(filename)
        self.word_tier = get_word_tier(self.textgrid)
        self.segment_tier = get_segment_tier(self.textgrid)
        self.set_info()

    def _add_words(self):
        self.words = []
        for index, interval in enumerate(self.word_tier.entries):
            self.words.append(Word(index, interval, self))
        
    def _add_phonemes(self):
        self.phonemes = []
        for index, interval in enumerate(self.segment_tier.entries):
            self.phonemes.append(Phoneme(index, interval, self))

    def _link_words_and_phonemes(self):
        for word in self.words:
            for phoneme in self.phonemes:
                if word.contains(phoneme):
                    phoneme.word = word
                    if phoneme.word != None: print(word, phoneme)
                    word.phonemes.append(phoneme)

    def set_info(self):
        self._add_words()
        self._add_phonemes()

class Word:
    def __init__(self, index, interval, awd):
        self.index = index
        self.interval = interval
        self.start = interval.start
        self.end = interval.end
        self.label = interval.label
        self.awd = awd
        self.phonemes = []

    def __repr__(self):
        m = self.label + ' ' + str(self.start) + ' ' + str(self.end)
        m += ' ' + str(self.index) 
        return m

    def contains(self, phoneme):
        return self.start <= phoneme.start and self.end >= phoneme.end

class Phoneme:
    def __init__(self, index, interval, awd):
        self.index = index
        self.interval = interval
        self.start = interval.start
        self.end = interval.end
        self.label = interval.label
        self.awd = awd
        self.word = None
    
    def __repr__(self):
        m = self.label + ' ' + str(self.start) + ' ' + str(self.end)
        m += ' ' + str(self.index) 
        return m

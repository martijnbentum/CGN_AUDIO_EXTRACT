from progressbar import progressbar
from text.models import Textgrid, Component

def get_k_textgrids():
    component = Component.objects.get(name='k')
    return component.textgrid_set.all()

def make_k_sentences():
    textgrids = get_k_textgrids()
    sentences = []
    for textgrid in progressbar(textgrids):
        sentences.extend(textgrid_to_sentence(textgrid))
    return sentences

def make_k_words(sentences = None):
    words = []
    if not sentences: sentences = make_k_sentences()
    for sentence in progressbar(sentences):
        for i, word in enumerate(sentence.words):
            words.append(Word(word, sentence, i))
    return words

def make_k_phonemes(sentences = None):
    phonemes = []
    if not sentences: sentences = make_k_sentences()
    for sentence in progressbar(sentences):
        for i, phoneme in enumerate(sentence.phonemes):
            phonemes.append(Phoneme(phoneme, sentence, i))
    return phonemes

def save_k_phonemes(phonemes = None):
    header = 'audio_filename\tstart_time\tend_time\tduration\tphoneme'
    header += '\tprevious_phoneme\tnext_phoneme\tspeaker_id\toverlap'
    o = [header]
    if not phonemes: phonemes = make_k_phonemes()
    for phoneme in phonemes:
        o.append(phoneme.line)
    with open('../news_phonemes_zs.tsv','w') as f:
        f.write('\n'.join(o))

def save_k_words(words = None):
    header = 'audio_filename\tstart_time\tend_time\tduration\ttext\tspeaker_id'
    header += '\toverlap'
    o = [header]
    if not words: words = make_k_words()
    for word in words:
        o.append(word.line)
    with open('../news_words_zs.tsv','w') as f:
        f.write('\n'.join(o))

def save_k_sentences(sentences = None):
    header = 'audio_filename\tstart_time\tend_time\tduration\ttext\tidentifier'
    header += '\tspeaker_ids\tin_pretraining'
    o = [header]
    if not sentences: sentences = make_k_sentences()
    for sentence in sentences:
        o.append(sentence.line)
    with open('../news_sentences_zs.tsv','w') as f:
        f.write('\n'.join(o))
    


def textgrid_to_sentence(textgrid):
    sentences = []
    temp = []
    index = 0
    for word in textgrid.word_set.all():
        if word.eos:
            temp.append(word)
            sentence = Sentence(temp, textgrid, index)
            sentences.append(sentence)
            temp = []
            index += 1
        else:
            temp.append(word)
    return sentences

def check_all_sentences_in_pretraining(sentences, cgn_speakers):
    for sentence in progressbar(sentences):
        check_sentence_in_pretraining(sentence, cgn_speakers)
    

def check_sentence_in_pretraining(sentence, cgn_speakers):
    speakers = sentence.textgrid.speakers.all()
    speaker_ids = [speaker.speaker_id for speaker in speakers]
    audio_filename = sentence.audio_filename
    for speaker_id in speaker_ids:
        if speaker_id not in cgn_speakers: 
            raise ValueError('Speaker not in cgn_speakers', speaker_id)
        for phrase in cgn_speakers[speaker_id]['phrases']:
            if phrase['audio_filename'] == audio_filename:
                sentence.in_pretraining = True
                return True
    sentence.in_pretraining = False
    return False

class Sentence:
    def __init__(self, words, textgrid, index):
        self.words = words
        self.start_time = words[0].start_time
        self.end_time = words[-1].end_time
        self.textgrid = textgrid
        self.index = index
        self.audio = textgrid.audio
        self.audio_filename = self.audio.filename.split('CGN2/')[-1]
        self.identifier = self.audio_filename.split('/')[-1].split('.')[-2] 
        self.identifier += '_sentence-' + str(self.index) + '.wav'
        self.duration = self.end_time - self.start_time
        speakers = textgrid.speakers.all()
        self.speaker_ids = ','.join([str(s).replace(' ','_') for s in speakers])
        self.sentence = ' '.join([w.awd_word for w in words])
        self.in_pretraining = None

    @property
    def phonemes(self):
        if hasattr(self, '_phonemes'): return self._phonemes
        output = []
        for word in self.words:
            awd_line = word.awd_phonemes
            d = eval(awd_line)
            for item in d.values(): 
                phoneme, start_time, end_time = item.split('\t')
                output.append(AWD(phoneme, start_time, end_time, word))
        self._phonemes = output
        return self._phonemes
            

    def __repr__(self):
        m = f'{self.identifier} {self.speaker_ids} {self.duration:.3f}'
        return m

    @property
    def line(self):
        m = f'{self.audio_filename}'
        m += f'\t{self.start_time:.3f}\t{self.end_time:.3f}'
        m += f'\t{self.duration:.3f}\t"{self.sentence}"' 
        m += f'\t{self.identifier}\t{self.speaker_ids}\t{self.in_pretraining}' 
        return m

class Word:
    def __init__(self, word, sentence, index):
        self.word = word
        self.sentence = sentence
        self.index = index
        self.filename = sentence.identifier
        self.start_time = word.start_time - sentence.start_time
        self.end_time = word.end_time - sentence.start_time
        self.duration = self.end_time - self.start_time
        self.speaker_id = str(word.speaker).replace(' ','_')

    def __repr__(self):
        m = f'{self.filename} {self.speaker_id} {self.duration:.3f}' 
        m += f' {self.word.awd_word}'
        return m

    @property
    def line(self):
        m = f'{self.filename}'
        m += f'\t{self.start_time:.3f}\t{self.end_time:.3f}'
        m += f'\t{self.duration:.3f}\t{self.word.awd_word}'
        m += f'\t{self.speaker_id}\t{self.word.overlap}'
        return m

class Phoneme:
    def __init__(self, phoneme, sentence, index):
        self.phoneme = phoneme
        self.phoneme_char = phoneme.phoneme
        self.start_time = phoneme.start_time - sentence.start_time
        self.end_time = phoneme.end_time - sentence.start_time
        self.duration = self.end_time - self.start_time
        self.sentence = sentence
        self.index = index
        self.filename = sentence.identifier
        self.start_time = phoneme.start_time - sentence.start_time
        self.end_time = phoneme.end_time - sentence.start_time
        self.duration = self.end_time - self.start_time
        self.speaker_id = str(phoneme.word.speaker).replace(' ','_')
        self.overlap = phoneme.word.overlap

    @property
    def previous_phoneme(self):
        if self.index == 0: return 'SOS'
        return self.sentence.phonemes[self.index - 1].phoneme

    @property
    def next_phoneme(self):
        if self.index == len(self.sentence.phonemes) - 1: return 'EOS'
        return self.sentence.phonemes[self.index + 1].phoneme

    @property
    def line(self):
        m = f'{self.filename}'
        m += f'\t{self.start_time:.3f}\t{self.end_time:.3f}'
        m += f'\t{self.duration:.3f}\t{self.phoneme_char}'
        m += f'\t{self.previous_phoneme}\t{self.next_phoneme}'
        m += f'\t{self.speaker_id}\t{self.overlap}'
        return m


class AWD:
    def __init__(self, phoneme, start_time, end_time, word):
        self.phoneme= phoneme
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.duration = self.end_time - self.start_time
        self.word = word
        
    def __repr__(self):
        m = f'{self.phoneme} {self.duration:.3f}'
        return m

        
        

def get_k_words():
    textgrids = get_k_textgrids()
    words = []
    for textgrid in textgrids:
        words.extend(textgrid.word_set.all())
    return words

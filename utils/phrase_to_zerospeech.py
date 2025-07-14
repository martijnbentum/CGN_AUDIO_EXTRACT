'''need phoneme alignments for phrases
the filenames have speaker id audio id start time end time structure
{speaker_id}_{audio_id}_{start_time_seconds}__{start_time_miliseconds}-{end_time_seconds}__{end_time_miliseconds}.wav
this module is not used for zerospeech_k or zerospeech_ko
'''
from django.core.exceptions import ObjectDoesNotExist
import json
from pathlib import Path
from progressbar import progressbar
from text.models import Audio
from utils import phonemes

sampa =  phonemes.Sampa()


class Phrases:
    def __init__(self, manifest=None, n=None, save_json=False, save_tsv=True):
        if manifest is None:
            print('No manifest provided, loading default manifest.')
            header, manifest = load_manifest()
        self.manifest = manifest
        if n is None: n = len(manifest)
        self.n = n
        self._load_phrases()
        if save_json:
            self.save_json_phrases
            self.save_json_words
            self.save_json_phonemes
        if save_tsv:
            self.phrases_tsv
            self.words_tsv
            self.phonemes_tsv

    def _load_phrases(self):
        self.phrases = []
        self.error_phrases = []
        self.process_errors = []
        print(f'Creating phrases from manifest, processing {self.n} lines.')
        for line in progressbar(self.manifest[:self.n]):
            audio_filename = line[0]
            try: phrase = Phrase(audio_filename)
            except:
                self.process_errors.append(audio_filename)
                continue
            if phrase.ok:
                self.phrases.append(phrase)
            else:
                self.error_phrases.append(phrase)

    @property
    def to_phrase_dict_list(self):
        output = [x.to_dict for x in self.phrases]
        return output

    @property
    def to_word_dict_list(self):
        output = []
        for phrase in self.phrases:
            for word in phrase.words:
                output.append(word.to_dict)
        return output

    @property
    def to_phoneme_dict_list(self):
        output = []
        for phrase in self.phrases:
            for phoneme in phrase.phonemes:
                output.append(phoneme.to_dict)
        return output

    @property
    def phrases_tsv(self):
        output = [self.phrases[0].header if self.phrases else '']
        for phrase in self.phrases:
            output.append(phrase.line)
        with open(f'../phrases_{self.n}.tsv', 'w') as f:
            f.write('\n'.join(output))
        return '\n'.join(output)

    @property
    def words_tsv(self):
        output = [self.phrases[0].words[0].header if self.phrases else '']
        for phrase in self.phrases:
            for word in phrase.words:
                output.append(word.line)
        with open(f'../words_{self.n}.tsv', 'w') as f:
            f.write('\n'.join(output))
        return '\n'.join(output)

    @property
    def phonemes_tsv(self):
        output = [self.phrases[0].phonemes[0].header if self.phrases else '']
        for phrase in self.phrases:
            for phoneme in phrase.phonemes:
                output.append(phoneme.line)
        with open(f'../phonemes_{self.n}.tsv', 'w') as f:
            f.write('\n'.join(output))
        return '\n'.join(output)

    @property
    def save_json_phrases(self):
        with open(f'../phrases_{self.n}.json', 'w') as f:
            json.dump(self.to_phrase_dict_list, f, indent=4, ensure_ascii=False)

    @property
    def save_json_words(self):
        with open(f'../words_{self.n}.json', 'w') as f:
            json.dump(self.to_word_dict_list, f, indent=4, ensure_ascii=False)
        
    @property
    def save_json_phonemes(self):
        with open(f'../phonemes_{self.n}.json', 'w') as f:
            json.dump(self.to_phoneme_dict_list, f, indent=4, ensure_ascii=False)
        
        
    
class Phrase:
    def __init__(self, audio_filename):
        self.audio_filename = audio_filename
        self.path = Path(audio_filename)
        self.ok = True
        self.error_info = ''
        self._set_filename_info()
        self._get_words()
        self._get_phonemes()
        self.phrase = ' '.join([w.word.awd_word for w in self.words])
        if self.raw_words:
            self.component = self.raw_words[0].textgrid.component.name 
        else:
            self.component = None
        if not self.words:
            self.ok = False
            self.error_info += 'No words found in phrase.\n'
        if not self.phonemes:
            self.ok = False
            self.error_info += 'No phonemes found in phrase.\n'

    def __repr__(self):
        m =  f'Phrase {self.audio_id} ' 
        m += f'({self.start_time:.1f} - {self.end_time:.1f}), '
        m += f'{self.duration:.1f} | {self.n_words} words, '
        m += f'{self.n_phonemes} phonemes | '
        m += f'{self.path.stem}' 
        return m

    def _set_filename_info(self):
        f = self.audio_filename
        speaker_id, audio_id, start_time, end_time = audio_filename_to_info(f)
        self.speaker_id = speaker_id
        self.audio_id = audio_id
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time

    def _get_words(self):
        self.raw_words = load_words_with_identifier(self.audio_id)
        self.words = []
        self.error_words = []
        phrase_word_index = 0
        for audio_word_index, word in enumerate(self.raw_words):
            if word.end_time < self.start_time or word.start_time > self.end_time:
                continue
            w = Word(word, audio_word_index, phrase_word_index, self)
            if w.ok:
                self.words.append(w)
                phrase_word_index += 1
            else: self.error_words.append(w)
        self.n_words = len(self.words)
        self.n_error_words = len(self.error_words)

    def _get_phonemes(self):
        self.phonemes = []
        self.error_phonemes = []
        for word in self.words:
            for phoneme_index in range(len(word.phoneme_dict)):
                phoneme = Phoneme(word, phoneme_index, self)
                if phoneme.ok:
                    self.phonemes.append(phoneme)
                else:
                    self.error_phonemes.append(phoneme)
        self.n_phonemes = len(self.phonemes)
        self.n_error_phonemes = len(self.error_phonemes)

    @property
    def overlap(self):
        return any(word.word.overlap for word in self.words)

    @property
    def identifier(self):
        return self.path.stem

    @property
    def line(self):
        m = f'{self.audio_filename}\t{self.audio_id}'
        m += f'\t{self.start_time:.3f}\t{self.end_time:.3f}'
        m += f'\t{self.duration:.3f}\t"{self.phrase}"' 
        m += f'\t{self.identifier}\t{self.speaker_id}\t{self.overlap}' 
        m += f'\t{self.component}'
        return m

    @property
    def header(self):
        m = 'audio_filename\taudio_id\tstart_time\tend_time\tduration\tphrase'
        m += '\tidentifier\tspeaker_id\toverlap\tcomponent'
        return m

    @property
    def to_dict(self):
        d = {}
        for column_name in self.header.split('\t'):
            d[column_name] = getattr(self, column_name)
        return d
            

class Word:
    def __init__(self, word, audio_word_index, phrase_word_index, phrase):
        self.word = word
        self.audio_word_index = audio_word_index
        self.phrase_word_index = phrase_word_index
        self.phrase = phrase
        self.raw_start_time = word.start_time
        self.raw_end_time = word.end_time
        self.start_time = self.raw_start_time - phrase.start_time
        self.end_time = self.raw_end_time - phrase.start_time
        self.duration = self.end_time - self.start_time
        self.error_info = ''
        self.ok = True
        try: self.phoneme_dict = eval(word.awd_phonemes)
        except AttributeError:
            self.ok = False
            self.error_info += 'no phoneme dictionary found for word.\n'
        if self.start_time < 0 or self.end_time > phrase.duration:
            self.ok = False
            self.error_info += 'Word start or end time is out of bounds.\n'

    def __repr__(self):
        return f'{self.word.awd_word} {self.start_time:.3f} {self.end_time:.3f}'

    @property
    def identifier(self):
        return self.phrase.identifier + f'_{self.phrase_word_index}'

    @property
    def phrase_identifier(self):
        return f'{self.phrase.identifier}'

    @property
    def audio_id(self):
        return self.phrase.audio_id

    @property
    def speaker_id(self):
        return self.phrase.speaker_id

    @property
    def audio_filename(self):
        return self.phrase.audio_filename

    @property
    def overlap(self):
        return self.word.overlap

    @property
    def line(self):
        m = f'{self.phrase_identifier}\t{self.audio_filename}'
        m += f'\t{self.start_time:.3f}\t{self.end_time:.3f}'
        m += f'\t{self.duration:.3f}\t{self.word.awd_word}'
        m += f'\t{self.audio_word_index}\t{self.phrase_word_index}'
        m += f'\t{self.audio_id}\t{self.speaker_id}\t{self.word.overlap}'
        return m

    @property
    def header(self):
        m = 'phrase_identifier\taudio_filename\tstart_time\tend_time'
        m += '\tduration\tword'
        m += '\taudio_word_index\tphrase_word_index'
        m += '\taudio_id\tspeaker_id\toverlap'
        return m
        
    @property
    def to_dict(self):
        d = {}
        for column_name in self.header.split('\t'):
            if column_name == 'word':
                d[column_name] = self.word.awd_word
            else:
                d[column_name] = getattr(self, column_name)
            if column_name in ['start_time', 'end_time', 'duration']:
                d[column_name] = round(d[column_name], 3)
        return d

class Phoneme:
    def __init__(self, word, phoneme_index, phrase):
        self.word = word
        self.audio_word_index = word.audio_word_index
        self.phrase_word_index = word.phrase_word_index
        self.phoneme_index = phoneme_index
        self.phoneme_dict = word.phoneme_dict
        self.phoneme_line = self.phoneme_dict[phoneme_index].split('\t')
        self.phrase = phrase
        self.ok = True
        self.error_info = ''
        self._set_info()

    def __repr__(self):
        if self.ok:
            return f'{self.ipa} {self.ok} {self.duration:.3f}' 
        else:
            return f'Error: {self.error_info}'

    def _set_info(self):
        self.raw_sampa = self.phoneme_line[0]
        self.raw_start_time = float(self.phoneme_line[1])
        self.raw_end_time = float(self.phoneme_line[2])
        self.duration = self.raw_end_time - self.raw_start_time
        try:
            self.sampa = sampa.to_simple_sampa_dict[self.raw_sampa] 
            self.ipa = sampa.to_simple_ipa_dict[self.raw_sampa]
        except KeyError:
            self.ok = False
            m = f'Phoneme {self.raw_sampa} not found in Sampa dictionary. '
            self.error_info += m
        self.start_time = self.raw_start_time - self.phrase.start_time
        self.end_time = self.raw_end_time - self.phrase.start_time


    @property
    def identifier(self):
        return self.phrase.identifier + f'_{self.word_phoneme_index}'

    @property
    def phrase_identifier(self):
        return f'{self.phrase.identifier}'

    @property
    def audio_id(self):
        return self.phrase.audio_id

    @property
    def speaker_id(self):
        return self.phrase.speaker_id

    @property
    def audio_filename(self):
        return self.phrase.audio_filename

    @property
    def overlap(self):
        return self.word.overlap

    @property
    def line(self):
        m = f'{self.phrase_identifier}\t{self.audio_filename}'
        m += f'\t{self.start_time:.3f}\t{self.end_time:.3f}'
        m += f'\t{self.duration:.3f}\t{self.word.word.awd_word}'
        m += f'\t{self.sampa}\t{self.ipa}'
        m += f'\t{self.phoneme_index}'
        m += f'\t{self.audio_word_index}\t{self.phrase_word_index}'
        m += f'\t{self.audio_id}\t{self.speaker_id}\t{self.word.overlap}'
        return m

    @property
    def header(self):
        m = 'phrase_identifier\taudio_filename\tstart_time\tend_time'
        m += '\tduration\tword\tsampa\tipa\tword_phoneme_index'
        m += '\taudio_word_index\tphrase_word_index'
        m += '\taudio_id\tspeaker_id\toverlap'
        return m

    @property
    def to_dict(self):
        d = {}
        for column_name in self.header.split('\t'):
            if column_name == 'word':
                d[column_name] = self.word.word.awd_word
            elif column_name == 'word_phoneme_index':
                d[column_name] = self.phoneme_index
            else:
                d[column_name] = getattr(self, column_name)
            if column_name in ['start_time', 'end_time', 'duration']:
                d[column_name] = round(d[column_name], 3)
        return d
        

def load_manifest():
    print('Loading manifest from ../manifest-831h-nl.tsv')
    with open('../manifest-831h-nl.tsv', 'r') as f:
        lines = [x.split('\t') for x in f.read().split('\n') if x]
    header = lines[0]
    data = lines[1:]
    return header, data

def load_audio_with_identifier(identifier):
    try: audio = Audio.objects.get(cgn_id=identifier)
    except ObjectDoesNotExist:
        print(f"Audio with identifier '{identifier}' does not exist.")
        return None
    return audio

def load_textgrid_with_identifier(identifier):
    audio = load_audio_with_identifier(identifier)
    textgrid = audio.textgrid_set.first()
    return textgrid

def load_words_with_identifier(identifier):
    textgrid = load_textgrid_with_identifier(identifier)
    if not textgrid:
        print(f"No textgrid found for identifier '{identifier}'.")
        return []
    words = textgrid.word_set.all()
    return list(words)

def load_phonemes_with_identifier(identifier):
    words = load_words_with_identifier(identifier)
    if not words:
        print(f"No words found for identifier '{identifier}'.")
        return []

def audio_filename_to_info(audio_filename):
    # {speaker_id}_{audio_id}_{start_time_seconds}__{start_time_miliseconds}-{end_time_seconds}__{end_time_miliseconds}.wav
    p = Path(audio_filename)
    temp = p.stem.split('_')
    speaker_id, audio_id, time_info = temp[0], temp[1], '_'.join(temp[2:])
    start_time, end_time = time_info.split('-')
    start_time = float(start_time.replace('__', '.'))
    end_time = float(end_time.replace('__', '.'))
    return speaker_id, audio_id, start_time, end_time
        


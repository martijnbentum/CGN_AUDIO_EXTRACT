import json
from . import phonemes
from progressbar import progressbar
from text.models import Speaker 

class CGN:
    def __init__(self, save = False):
        self.speakers = Speaker.objects.all()
        self._make_speaker_phrases()
        if save: self.to_json

    def _make_speaker_phrases(self):
        self.d = {}
        for speaker in progressbar(self.speakers):
            sp = SpeakerPhrases(speaker)
            self.d[speaker.speaker_id] = sp.to_dict
            
    @property
    def to_dict(self):
        return self.d

    @property
    def to_json(self):
        with open('../cgn.json','w') as f:
            json.dump(self.d,f)


class SpeakerPhrases:
    def __init__(self, speaker):
        self.speaker= speaker
        self.speaker_id = speaker.speaker_id
        self.textgrids = speaker.textgrid_set.all()
        self._make_phrases()

    def _make_phrases(self):
        self.phrases = []
        for textgrid in self.textgrids:
            self._handle_textgrid(textgrid)

    def _handle_textgrid(self, textgrid):
        d = textgrid.speaker_to_phrases_dict(end_on_eos = False, 
            maximum_duration = 15)
        phrases = d[self.speaker_id]
        for phrase in phrases:
            self.phrases.append(Phrase(phrase))

    @property
    def to_dict(self):
        d = {}
        d['duration'] = sum([x.duration for x in self.phrases])
        d['phrases'] = [p.to_dict for p in self.phrases]
        d['language'] = self.phrases[0].language
        return d


class Phrase:
    def __init__(self, words):
        self.words = words
        self._set_info()

    def _set_info(self):
        self.start_time = self.words[0].start_time
        self.end_time = self.words[-1].end_time
        self.duration = self.end_time - self.start_time
        self._set_file_info()
        self._set_audio_info()
        self._set_speaker_info()
        self._set_orthographic_info()
        self._set_phonemic_info()
        self._set_language_info()

    def _set_file_info(self):
        self.textgrid = self.words[0].textgrid
        self.cgn_id = self.textgrid.cgn_id
        self.component = self.textgrid.component.name
        f = self.textgrid.audio.filename
        self.audio_filename = f.replace('/vol/bigdata/corpora2/CGN2/', '')

    def _set_audio_info(self):
        self.nchannels = self.textgrid.audio.nchannels
        self.sample_rate = self.textgrid.audio.sample_rate
        self.nspeakers_in_file = self.textgrid.nspeakers
        self.duration_source_file = self.textgrid.audio.duration

    def _set_speaker_info(self):
        self.speaker_id = self.words[0].speaker.speaker_id
        self.gender = self.words[0].speaker.gender
        self.age = self.words[0].speaker.age

    def _set_orthographic_info(self):
        self.orthographic = ' '.join([w.awd_word for w in self.words])

    def _set_phonemic_info(self):
        self.raw_phonemic = ' '.join([w.awd_word_phoneme for w in self.words])
        t = self.raw_phonemic
        for char in '#_-[]=':
            t = t.replace(char,'')
        self.clean_phonemic = t.strip()
        t = self.clean_phonemic
        d = phonemes.Sampa().to_simple_sampa_dict
        for sampa_symbol, simple_sampa_symbol in d.items():
            if sampa_symbol == simple_sampa_symbol: continue
            t = t.replace(sampa_symbol,simple_sampa_symbol)
        self.sampa = t.strip()

    def _set_language_info(self):
        if '/comp-' + self.component + '/nl/' in self.audio_filename:
            self.language = 'Netherlandic Dutch'
        if '/comp-' + self.component + '/vl/' in self.audio_filename:
            self.language = 'Flemish Dutch'

    @property
    def to_dict(self):
        keys = ['audio_filename','start_time', 'end_time', 'duration', 
            'cgn_id', 'component', 'nchannels', 'sample_rate', 
            'nspeakers_in_file', 'duration_source_file', 'speaker_id',
            'age','gender','orthographic','sampa', 'language']
        d = {}
        for key in keys:
            d[key] = getattr(self,key)
        return d


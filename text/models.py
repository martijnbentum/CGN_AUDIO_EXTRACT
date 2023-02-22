from django.db import models
import textgrids
from utils import word, phonemes


# Create your models here.

class Component(models.Model):
    name= models.CharField(max_length=1,unique = True)
    
    def __str__(self):
        return self.name

class Audio(models.Model):
    dargs = {'on_delete':models.SET_NULL,'blank':True,'null':True}
    cgn_id = models.CharField(max_length=30,unique = True)
    filename = models.CharField(max_length=1000,blank=True,null=True) 
    nchannels = models.PositiveIntegerField(null=True,blank=True)
    sample_rate = models.PositiveIntegerField(null=True,blank=True)
    duration = models.PositiveIntegerField(null=True,blank=True)
    component= models.ForeignKey(Component,**dargs)

    def __str__(self):
        m = self.cgn_id + ' ' 
        m += str(self.component.name) + ' ' 
        m += str(self.nchannels) + ' ' 
        m += str(self.duration)
        return m

class Speaker(models.Model):
    dargs = {'on_delete':models.SET_NULL,'blank':True,'null':True}
    speaker_id = models.CharField(max_length=30,unique = True)
    gender = models.CharField(max_length=10, blank=True, null=True)
    information = models.CharField(max_length=300, blank=True, null=True)
    birth_year = models.PositiveIntegerField(null=True,blank=True)
    age = models.PositiveIntegerField(null=True,blank=True)

    def __str__(self):
        m = self.speaker_id + ' '
        m += self.gender 
        if self.age:
            m += ' ' + str(self.age)
        return m

class Textgrid(models.Model):
    dargs = {'on_delete':models.SET_NULL,'blank':True,'null':True}
    cgn_id = models.CharField(max_length=30,blank=True, default=None)
    awd_filename = models.CharField(max_length=300,blank=True, default= None)
    fon_filename = models.CharField(max_length=300,blank=True, null=True)
    ort_filename = models.CharField(max_length=300,blank=True, null=True)
    speakers = models.ManyToManyField(Speaker,blank=True, default= None) 
    audio = models.ForeignKey(Audio,**dargs)
    component= models.ForeignKey(Component,**dargs)
    nspeakers = models.PositiveIntegerField(null=True,blank=True)

    def __str__(self):
        m = self.cgn_id + ' '
        m += ' | comp: ' + self.component.name 
        m += ' | dur: ' + str(self.audio.duration) 
        m += ' | nspeakers: ' + str(self.nspeakers)
        return m

    @property
    def speakers_str(self):
        return ' | '.join([str(x) for x in self.speakers.all()])

    def load_awd(self):
        return textgrids.TextGrid(self.awd_filename)

    def load_fon(self):
        return textgrids.TextGrid(self.fon_filename)

    def load_ort(self):
        return textgrids.TextGrid(self.ort_filename)

    def speaker_to_words_dict(self):
        if hasattr(self,'_speaker_to_words_dict'): 
            return self._speaker_to_words_dict
        d = {}
        for speaker in self.speakers.all():
            d[speaker.speaker_id] = self.word_set.filter(speaker = speaker)
        self._speaker_to_words_dict = d
        return self._speaker_to_words_dict  
                     
    def speaker_to_phrases_dict(self, end_on_eos = True, minimum_duration=None,
        maximum_duration=None, fon_alphabet = 'simple_json'):
        d = self.speaker_to_words_dict()
        od = {}
        for speaker_id, words in d.items():
            od[speaker_id] = word.words_to_phrases(words,end_on_eos,
                minimum_duration,maximum_duration)
        self._speaker_words_to_phrases_dict = od
        return self._speaker_words_to_phrases_dict


    def phrases(self, end_on_eos = True, minimum_duration = None,
        maximum_duration= None):
        '''phrases created based on awd transcribed words'''
        d = self.speaker_to_phrases_dict(end_on_eos,minimum_duration,
            maximum_duration)
        o = []
        for phrases in d.values():
            o.extend(phrases)
        return o

    def fon_phrases(self, minimum_duration = None,maximum_duration= None):
        '''phrases based on manually phonetically transcribed fon files'''
        p = self.fon_phrase_set.all()
        if minimum_duration:
            p = [x for x in p if x.duration > minimum_duration]
        if maximum_duration:
            p = [x for x in p if x.duration < maximum_duration]
        return p
            
class Fon_phrase(models.Model):
    dargs = {'on_delete':models.SET_NULL,'blank':True,'null':True}
    phrase = models.CharField(max_length=500,default=None,null=True) 
    overlap = models.BooleanField(default=False)
    start_time = models.FloatField(default = None,null=True)
    end_time = models.FloatField(default = None,null=True)
    textgrid= models.ForeignKey(Textgrid,**dargs)
    speaker= models.ForeignKey(Speaker,**dargs)
    fon_tier_index = models.PositiveIntegerField(null=True,blank=True)
    
    def __str__(self):
        m = self.phrase[:60] 
        if len(self.phrase) > 60: m += ' ...'
        m += ' | ' + str(round(self.end_time - self.start_time,2))
        return m

    @property
    def phrase_clean(self):
        if hasattr(self, '_phrase_clean'): return self._phrase_clean
        t = self.phrase
        for char in '#_-[]':
            t = t.replace(char,'')
        self._phrase_clean = t.strip()
        return self._phrase_clean

    @property
    def phrase_simple_sampa(self):
        if hasattr(self, '_phrase_simple_sampa'): 
            return self._phrase_simple_sampa
        t = self.phrase_clean
        d = phonemes.Sampa().to_simple_sampa_dict
        for sampa_symbol, simple_sampa_symbol in d.items():
            if sampa_symbol == simple_sampa_symbol: continue
            t = t.replace(sampa_symbol,simple_sampa_symbol)
        self._phrase_simple_sampa= t.strip()
        return self._phrase_simple_sampa
        
    @property
    def phrase_ort(self):
        if hasattr(self, '_phrase_ort'): return self._phrase_ort
        ort = self.textgrid.load_ort()
        return ort[self.speaker.speaker_id][self.fon_tier_index].text

    @property
    def duration(self):
        return self.end_time - self.start_time

    
    

class Word(models.Model):
    dargs = {'on_delete':models.SET_NULL,'blank':True,'null':True}
    awd_word = models.CharField(max_length=100,default=None,null=True)
    awd_word_phoneme = models.CharField(max_length=100,default=None,null=True)
    awd_phonemes = models.TextField(default='')
    start_time = models.FloatField(default = None,null=True)
    end_time = models.FloatField(default = None,null=True)
    textgrid= models.ForeignKey(Textgrid,**dargs)
    speaker= models.ForeignKey(Speaker,**dargs)
    awd_word_tier_index = models.PositiveIntegerField(null=True,blank=True)
    overlap = models.BooleanField(default=False)
    special_word= models.BooleanField(default=False)
    eos= models.BooleanField(default=False) 

    def __str__(self):
        m = self.awd_word + ' | ' + self.awd_word_phoneme
        m += ' | ' + str(round(self.end_time - self.start_time,2))
        return m

    def __eq__(self, other):
        if not type(self) == type(other): return False
        return self.pk == other.pk
        


    class Meta:
        unique_together = ('textgrid','speaker','awd_word_tier_index')

    @property
    def sampa_phonemes(self):
        p = eval(self.awd_phonemes).values()
        return [x.split('\t')[0].strip('=') for x in p]

    @property
    def duration(self):
        return self.end_time - self.start_time


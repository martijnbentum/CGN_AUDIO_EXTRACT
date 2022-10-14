from django.db import models
import textgrids
from utils import word


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

    def speaker_to_words_dict(self):
        if hasattr(self,'_speaker_to_words_dict'): 
            return self._speaker_to_words_dict
        d = {}
        for speaker in self.speakers.all():
            d[speaker.speaker_id] = self.word_set.filter(speaker = speaker)
        self._speaker_to_words_dict = d
        return self._speaker_to_words_dict  
                     
    def speaker_to_phrases_dict(self, end_on_eos = True, minimum_duration=None,
        maximum_duration=None):
        d = self.speaker_to_words_dict()
        od = {}
        for speaker_id, words in d.items():
            od[speaker_id] = word.words_to_phrases(words,end_on_eos,
                minimum_duration,maximum_duration)
        self._speaker_words_to_phrases_dict = od
        return self._speaker_words_to_phrases_dict


    def phrases(self, end_on_eos = True, minimum_duration = None,
        maximum_duration= None):
        d = self.speaker_to_phrases_dict(end_on_eos,minimum_duration,
            maximum_duration)
        o = []
        for phrases in d.values():
            o.extend(phrases)
        return o
            

class Word(models.Model):
    dargs = {'on_delete':models.SET_NULL,'blank':True,'null':True}
    word = models.CharField(max_length=100,default=None,null=True)
    word_phoneme = models.CharField(max_length=100,default=None,null=True)
    phonemes = models.TextField(default='')
    word_ipa_phoneme = models.CharField(max_length=100,default=None,null=True)
    ipa = models.CharField(max_length=100,default=None,null=True)
    start_time = models.FloatField(default = None,null=True)
    end_time = models.FloatField(default = None,null=True)
    textgrid= models.ForeignKey(Textgrid,**dargs)
    speaker= models.ForeignKey(Speaker,**dargs)
    awd_word_tier_index = models.PositiveIntegerField(null=True,blank=True)
    overlap = models.BooleanField(default=False)
    special_word= models.BooleanField(default=False)
    eos= models.BooleanField(default=False) 

    def __str__(self):
        m = self.word + ' | ' + self.word_phoneme
        m = self.word + ' | ' + self.word_ipa_phoneme
        m += ' | ' + str(round(self.end_time - self.start_time,2))
        return m

    class Meta:
        unique_together = ('textgrid','speaker','awd_word_tier_index')

    @property
    def sampa_phonemes(self):
        p = eval(self.phonemes).values()
        return [x.split('\t')[0].strip('=') for x in p]

    @property
    def duration(self):
        return self.end_time - self.start_time


from django.db import models

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

class Textgrid(model.Model):
    dargs = {'on_delete':models.SET_NULL,'blank':True,'null':True}
    cgn_id = models.CharField(max_length=30,blank=True, default=None)
    awd_filename = models.CharField(max_length=300,blank=True, default= None)
    speakers = models.ManyToManyField(Speaker,blank=True, default= None) 
    audio = models.ForeignKey(Audio,**dargs)
    component= models.ForeignKey(Component,**dargs)

class Word(models.Model):
    dargs = {'on_delete':models.SET_NULL,'blank':True,'null':True}
    word = models.CharField(max_length=100,default=None,null=True)
    word_phoneme = models.CharField(max_length=100,default=None,null=True)
    start_time = models.FloatField(default = None,null=True)
    end_time = models.FloatField(default = None,null=True)
    textgrid= models.ForeignKey(Textgrid,**dargs)
    speaker= models.ForeignKey(Speaker,**dargs)
    awd_word_tier_index = models.PositiveIntegerField(null=True,blank=True)



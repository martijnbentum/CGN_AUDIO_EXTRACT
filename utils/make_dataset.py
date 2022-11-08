from utils import identifiers
from utils import phonemes
import json
import os
import random

cache_dir = '../WAV2VEC_DATA/'
json_dir = cache_dir + 'JSONS/'
vocab_filename = cache_dir + 'vocab.json'
cgn_ids_tdt_split_filename = cache_dir + 'cgn_ids_tdt_split.json'
fon_cgn_ids_tdt_split_filename = cache_dir + 'fon_cgn_ids_tdt_split.json'

def make_vocab_dict(save = False):
    sampa = phonemes.Sampa()
    p= [s for s in sampa.simple_symbols]
    p = list(set(p))
    p.append(' ')
    d = {v: k for k,v in enumerate(sorted(p))}
    d['[UNK]'] = len(d)
    d['[PAD]'] = len(d)
    d['|'] = d[' ']
    del d[' ']
    if save:
        with open(vocab_filename,'w') as fout:
            json.dump(d,fout)
    return d
    

def _make_fon_cgn_id_train_dev_test_split(train_perc = .8, save = False,seed=9,
    exclude_components = ['m']):
    '''create a list of cgn_ids split in train dev test set per component.
    and an all for all components
    this only uses the files with a manual phonetic transcription
    train_perc      the percentage of materials used for training
    save            whether to save the split
    seed            change this is you want to have a new random shuffle of
                    the data
    '''
    random.seed(seed)
    comp_cgn_id = identifiers.component_to_cgn_ids_dict()
    d ={'all':{'train':[],'dev':[],'test':[]}}
    for component,cgn_ids in comp_cgn_id.items():
        if component in exclude_components:
            print('exclude component:',component)
            continue
        d[component] = {}
        random.shuffle(cgn_ids)
        train_index = int(len(cgn_ids) * train_perc)
        dev_index = int(len(cgn_ids) * (train_perc +(1 - train_perc)/2) )
        print(component,len(cgn_ids),train_index,dev_index)
        d[component]['train'] = cgn_ids[:train_index]
        d[component]['dev'] = cgn_ids[train_index:dev_index]
        d[component]['test'] = cgn_ids[dev_index:]
        d['all']['train'].extend(cgn_ids[:train_index])
        d['all']['dev'].extend(cgn_ids[train_index:dev_index])
        d['all']['test'].extend(cgn_ids[dev_index:])
    if save:
        with open(fon_cgn_ids_tdt_split_filename,'w') as fout:
            json.dump(d,fout)
    return d

def _make_cgn_id_train_dev_test_split(train_perc = .8, save = False,seed=9,
    exclude_components = ['m']):
    '''create a list of cgn_ids split in train dev test set per component.
    and an all for all components
    this is based on awd (automatically transcribed files)
    train_perc      the percentage of materials used for training
    save            whether to save the split
    seed            change this is you want to have a new random shuffle of
                    the data
    '''
    random.seed(seed)
    comp_cgn_id = identifiers.component_to_cgn_ids_dict()
    d ={'all':{'train':[],'dev':[],'test':[]}}
    for component,cgn_ids in comp_cgn_id.items():
        if component in exclude_components:
            print('exclude component:',component)
            continue
        d[component] = {}
        random.shuffle(cgn_ids)
        train_index = int(len(cgn_ids) * train_perc)
        dev_index = int(len(cgn_ids) * (train_perc +(1 - train_perc)/2) )
        print(component,len(cgn_ids),train_index,dev_index)
        d[component]['train'] = cgn_ids[:train_index]
        d[component]['dev'] = cgn_ids[train_index:dev_index]
        d[component]['test'] = cgn_ids[dev_index:]
        d['all']['train'].extend(cgn_ids[:train_index])
        d['all']['dev'].extend(cgn_ids[train_index:dev_index])
        d['all']['test'].extend(cgn_ids[dev_index:])
    if save:
        with open(cgn_ids_tdt_split_filename,'w') as fout:
            json.dump(d,fout)
    return d


def load_fon_cgn_id_train_dev_test_split():
    '''load the dictionary containing cgn ids split into train dev and test
    sets.
    the sets are split up per component and one all combining the components
    '''
    if not os.path.isfile(fon_cgn_ids_tdt_split_filename):
        _make_fon_cgn_id_train_dev_test_split(save=True)
    with open(cgn_ids_tdt_split_filename) as fin:
        d = json.load(fin)
    return d
    
def load_cgn_id_train_dev_test_split():
    '''load the dictionary containing cgn ids split into train dev and test
    sets.
    the sets are split up per component and one all combining the components
    '''
    if not os.path.isfile(cgn_ids_tdt_split_filename):
        _make_cgn_id_train_dev_test_split(save=True)
    with open(cgn_ids_tdt_split_filename) as fin:
        d = json.load(fin)
    return d

def phrase_to_sentence(phrase, change_phonemes = None):
    sentence = ' '.join([word.word_ipa_phoneme for word in phrase])
    if type(change_phonemes) == dict:
        for old_phone,new_phone in change_phonemes.items():
            sentence = sentence.replace(old_phone,new_phone)
    return sentence

def fon_phrase_to_dict(fon_phrase,  minimal = True):
    audio = fon_phrase.textgrid.audio
    speaker = fon_phrase.speaker
    d = {}
    d['sentence'] = fon_phrase.phrase_simple_sampa
    d['start_time'] = fon_phrase.start_time
    d['end_time'] = fon_phrase.end_time
    d['audiofilename'] = audio.filename
    d['sampling_rate'] = audio.sample_rate
    if minimal: return d
    d['speaker'] = speaker.speaker_id
    d['component'] = audio.component.name
    return d

def phrase_to_dict(phrase, change_phonemes = None, minimal = True):
    audio = phrase[0].textgrid.audio
    speaker = phrase[0].speaker
    d = {}
    d['sentence'] = phrase_to_sentence(phrase, change_phonemes)
    d['start_time'] = phrase[0].start_time
    d['end_time'] = phrase[-1].end_time
    d['audiofilename'] = audio.filename
    d['sampling_rate'] = audio.sample_rate
    if minimal: return d
    d['speaker'] = speaker.speaker_id
    d['component'] = audio.component.name
    return d


def _handle_fon_cgn_ids(cgn_ids, minimum_duration, maximum_duration):
    from text.models import Textgrid
    data = []
    for i,cgn_id in enumerate(cgn_ids):
        try: tg = Textgrid.objects.get(cgn_id = cgn_id)
        except Textgrid.DoesNotExist: 
            print('can not load textgrid with id:',cgn_id)
            return
        if i % 100 == 0:print('handling:',cgn_id,i,len(cgn_ids))
        phrases = tg.fon_phrases(
            minimum_duration=minimum_duration,
            maximum_duration= maximum_duration)
        for phrase in phrases:
            d = fon_phrase_to_dict(phrase)
            if not d['sentence']: continue
            data.append( d )
    return data
    
def _handle_cgn_ids(cgn_ids, minimum_duration, maximum_duration,
    change_phonemes = None):
    from text.models import Textgrid
    data = []
    for i,cgn_id in enumerate(cgn_ids):
        try: tg = Textgrid.objects.get(cgn_id = cgn_id)
        except Textgrid.DoesNotExist: 
            print('can not load textgrid with id:',cgn_id)
            return
        if i % 100 == 0:print('handling:',cgn_id,i,len(cgn_ids))
        phrases = tg.phrases(
            end_on_eos = False, 
            minimum_duration=minimum_duration,
            maximum_duration= maximum_duration)
        for phrase in phrases:
            data.append( phrase_to_dict(phrase, change_phonemes) )
    return data

def _save_json(d, filename):
    print('saving:',filename)
    with open(filename,'w') as fout:
        json.dump(d,fout)
            

def make_fon_jsons(minimum_duration= 0.9, maximum_duration = 7,
    save = False):
    '''create json files for train dev test sets based on fon files'''
    error = []
    datasets = {}
    for name, component in load_fon_cgn_id_train_dev_test_split().items():
        datasets[name] = {}
        for set_name, cgn_ids in component.items():
            filename = json_dir + name+'_'+set_name+ '.json'
            print('handling:',filename)
            data = _handle_fon_cgn_ids(cgn_ids,minimum_duration,maximum_duration)
            if not data: continue
            datasets[name][set_name] = {'data':data} 
            if save: _save_json({'data':data},filename)
    return datasets
        
def make_jsons(minimum_duration= 0.9, maximum_duration = 7,
    change_phonemes = {'ʒ':'zj','ɲ':'nj'}, save = False):
    '''create json files for train dev test sets'''
    error = []
    datasets = {}
    for name, component in load_cgn_id_train_dev_test_split().items():
        datasets[name] = {}
        for set_name, cgn_ids in component.items():
            filename = json_dir + name+'_'+set_name+ '.json'
            print('handling:',filename)
            data = _handle_cgn_ids(cgn_ids,minimum_duration,maximum_duration,
                change_phonemes) 
            if not data: continue
            datasets[name][set_name] = {'data':data} 
            if save: _save_json({'data':data},filename)
    return datasets
            
    
    

# part of the ASTA project to align wav2vec and cgn with nw

import glob
import json
import os
import random
from utils import needleman_wunch as nw
from text.models import Textgrid

cgn_wav2vec_dir = '/vol/tensusers3/mbentum/cgn_wav2vec/'
cgn_align = '/vol/tensusers3/mbentum/cgn_wav2vec_align/'

def table_filenames():
    fn = glob.glob(cgn_wav2vec_dir + '*.table')
    return fn

def text_filename():
    fn = glob.glob(cgn_wav2vec_dir + '*.txt')
    return fn

def load_table(filename):
    with open(filename) as fin:
        t = fin.read()
    temp= [x.split('\t') for x in t.split('\n') if x]
    table = []
    for grapheme, start, end in temp:
        table.append([grapheme,float(start), float(end)])
    return table

def load_text(filename):
    with open(filename) as fin:
        t = fin.read()
    return t

def table_filename_to_cgn_id(f):
    cgn_id = f.split('/')[-1].split('.')[0]
    return cgn_id

def make_alignments():
    fn = table_filenames()
    random.shuffle(fn)
    for f in fn:
        cgn_id = table_filename_to_cgn_id(f)
        align_filename = cgn_align + cgn_id
        if os.path.isfile(align_filename): 
            print('skiping',cgn_id)
            continue
        print('handling',cgn_id)
        Align(cgn_id, make_phrases = False)
        
    

    
class Align:
    '''align object to align cgn textgrid and wav2vec transcript.
    '''
    def __init__(self,cgn_id, make_phrases = True):
        self.cgn_id = cgn_id
        self.textgrid = Textgrid.objects.get(cgn_id = cgn_id)
        self.awd_words = list(self.textgrid.word_set.all())
        # self.awd_text = ' '.join([w.awd_word for w in self.awd_words])
        self.awd_text = phrases_to_text(self.textgrid.phrases())
        self._set_wav2vec_table_and_text()
        self._set_align()
        if make_phrases:
            self._set_phrases()

    def _set_wav2vec_table_and_text(self):
        self.wav2vec_base_filename = cgn_wav2vec_dir + self.textgrid.cgn_id
        table = load_table(self.wav2vec_base_filename+'.table')
        self.wav2vec_table = fix_unk_in_table(table)
        self.wav2vec_text = load_text(self.wav2vec_base_filename+'.txt')

    def _set_align(self):
        self.align_filename = cgn_align + self.cgn_id
        print(self.align_filename)
        if os.path.isfile(self.align_filename): 
            with open(self.align_filename) as fin:
                self.align = fin.read()
        else:
            self.align = nw.nw(self.awd_text, self.wav2vec_text)
            with open(self.align_filename, 'w') as fout:
                fout.write(self.align)
        
    def _set_phrases(self):
        phrases = sort_phrases(self.textgrid.phrases())
        o = align_phrases_with_aligned_text(phrases,self.aligned_cgn_text,
            self.awd_words)
        self.phrases = []
        for phrase, start_index, end_index in o:
            p = Phrase(phrase,self,start_index, end_index)
            self.phrases.append(p)
            
    def alignment_labels(self, delta = .5):
        return [p.alignment(delta = delta) for p in self.phrases]

    def perc_bad(self, delta = .5):
        labels = self.alignment_labels(delta)
        ntotal = len(labels)
        if ntotal == 0: return 0
        nbad = labels.count('bad')
        return round(nbad / ntotal * 100,2)

    @property
    def duration(self):
        return self.textgrid.audio.duration

    @property
    def component(self):
        return self.textgrid.component.name

    @property
    def nspeakers(self):
        return self.textgrid.nspeakers

    @property
    def aligned_wav2vec_text(self):
        return self.align.split('\n')[1]
        
    @property
    def aligned_cgn_text(self):
        return self.align.split('\n')[0]

    @property
    def wav2vec_aligned_graphemes_timestamps(self):
        o = []
        i = 0
        for char in self.aligned_wav2vec_text:
            if char == '-': 
                o.append([])
                continue
            # print(i,char)
            o.append(self.wav2vec_table[i])
            i += 1
        return o

    @property
    def average_match_perc(self):
        match_percs = []
        for phrase in self.phrases:
            match_percs.append(phrase.match_perc)
        if len(match_percs) == 0: return 0
        return round(sum(match_percs) / len(match_percs),2)


class Phrase:
    def __init__(self,phrase,align=None,start_index=None,end_index=None):
        self.phrase = phrase
        self.align = align
        self.start_time = self.phrase[0].start_time
        self.end_time = self.phrase[-1].end_time
        self.duration = self.end_time - self.start_time
        self.nwords = len(self.phrase)
        self.start_index = start_index
        self.end_index = end_index
        self._set_info()

    def __repr__(self):
        m = self.text[:18].ljust(21)
        m += ' | ' + self.alignment()
        return m

    def __str__(self):
        m = self.aligned_cgn_text + '\n'
        m += self.aligned_wav2vec_text + '\n'
        m += 'cgn ts:'.ljust(12) + str(round(self.start_time,2)) + ' '
        m += str(round(self.end_time,2)) + '\n' 
        m += 'w2v ts:'.ljust(12) + str(self.wav2vec_start_time) + ' '
        m += str(self.wav2vec_end_time) +'\n' 
        m += 'alignment:'.ljust(12) + self.alignment() + '\n'
        m += 'match:'.ljust(12) + str(self.match_perc)
        return m 

    def _set_info(self):
        if not self.align: return
        if self.start_index == self.end_index == None: return
        align = self.align
        start, end = self.start_index, self.end_index
        cgn = align.aligned_cgn_text[start:end]
        wav2vec= align.aligned_wav2vec_text[start:end]
        graphemes = align.wav2vec_aligned_graphemes_timestamps[start:end]
        self.aligned_cgn_text= cgn
        self.cgn_text = cgn.replace('-','')
        self.aligned_wav2vec_text= wav2vec
        self.wav2vec_text = wav2vec.replace('-','')
        self.wav2vec_aligned_graphemes = graphemes
        self._set_wav2vec_timestamps()

    def _set_wav2vec_timestamps(self):
        self.wav2vec_start_time= None
        self.wav2vec_end_time= None
        for line in self.wav2vec_aligned_graphemes:
            if len(line) == 3 and line[0] != ' ':
                self.wav2vec_start_time= line[1]
                break
        for line in self.wav2vec_aligned_graphemes[::-1]:
            if len(line) == 3 and line[0] != ' ':
                self.wav2vec_end_time= line[-1]
                break
        if self.wav2vec_start_time == self.wav2vec_end_time == None:
            self.wav2vec_timestamps_ok = False
        else:
            self.wav2vec_timestamps_ok = False

    def alignment(self,delta = 0.5):
        if self.start_index == self.end_index == None: return 'bad'
        text_ok = self.cgn_text == self.text
        d = delta
        start_ok=equal_with_delta(self.start_time,self.wav2vec_start_time,d)
        end_ok = equal_with_delta( self.end_time, self.wav2vec_end_time, d)
        if start_ok and end_ok: return 'good'
        if start_ok:  return 'start match'
        if end_ok:  return 'end match'
        if text_ok: return 'middle match'
        return 'bad'

    @property
    def match_perc(self):
        match = 0
        nchar = len(self.aligned_cgn_text)
        if nchar == 0: return 0
        texts = zip(self.aligned_cgn_text,self.aligned_wav2vec_text)
        for cgn_char,w2v_char in texts:
            if cgn_char == w2v_char: match += 1
        return round(match / nchar * 100,2)

    @property
    def text(self):
        return phrase_to_text(self.phrase)

    @property
    def nchars(self):
        return len(self.text)

            
def phrase_to_text(phrase):
    return ' '.join([w.awd_word for w in phrase])

def phrase_to_len(phrase):
    return len(phrase_to_text(phrase))

def _find_end_index(phrase_text, text, start_index, end_index):
    if type(start_index) != int: return False
    if end_index < start_index + len(phrase_text):
        end_index = start_index + len(phrase_text)
    compare_text = text[start_index:end_index].replace('-','').strip()
    #print(phrase_text, compare_text)
    if phrase_text.replace('-','') == compare_text:
        return end_index
    if end_index > len(text): return False
    return _find_end_index(phrase_text,text,start_index,end_index+1)
   

def align_phrases_with_aligned_text(phrases, text, word_list):
    output = []
    start = 0
    word_indices = []
    for phrase in phrases:
        if hasattr(phrase,'text'): phrase = phrase.phrase
        pt = phrase_to_text(phrase)
        end = _find_end_index(pt,text,start, start)
        if not end:
            output.append([phrase,None,None])
        else:
            output.append([phrase,start,end])
            start = end
    return output



    
def equal_with_delta(n1,n2,delta):
    if type(n1) != float: raise ValueError('gt timestamp should be available')
    if type(n2) != float: return False
    lower_bound, upper_bound = n1-delta, n1+delta
    if n2 >= lower_bound and n2 <= upper_bound: return True
    return False

def sort_phrases(phrases):
    return sorted(phrases, key = lambda x: x[0].start_time)

def phrases_to_text(phrases):
    p= sort_phrases(phrases)
    o = []
    for phrase in p:
        o.append(phrase_to_text(phrase))
    return ' '.join(o)

def _unk_in_table(table):
    for line in table:
        if line[0] == '[UNK]': return True
    return False

def fix_unk_in_table(table):
    if not _unk_in_table(table): return table
    output = []
    for line in table:
        if line[0] == '[UNK]':
            output.extend([[x,line[1],line[2]] for x in list('[UNK]')])
        else: output.append(line)
    return output
    

def extract_all_phrases(aligns):
    phrases = []
    for align in aligns:
        phrases.extend(align.phrases)
    return phrases

def phrase_to_dataset_line(phrase, phrase_index):
    p, a = phrase, phrase.align
    line = [phrase_index,a.cgn_id, a.duration, a.component,a.nspeakers]
    line.extend([p.start_index,p.end_index, p.nwords])
    line.extend([round(p.duration,2),p.start_time,p.end_time])
    line.extend([p.wav2vec_start_time,p.wav2vec_end_time])
    line.extend([p.alignment(),p.alignment(1),p.alignment(0.1)])
    line.append(p.match_perc)
    return line

def phrases_to_dataset(phrases):
    ds = []
    for i,phrase in enumerate(phrases):
        line = phrase_to_dataset_line(phrase,i)
        ds.append(line)
    return ds

def align_to_dataset_line(align):
    a = align
    line = [a.cgn_id, a.duration, a.component, a.nspeakers]
    line.extend( [len(a.phrases), a.average_match_perc, a.perc_bad()] )
    line.extend( [a.perc_bad(1),a.perc_bad(0.1)] )
    return line

def save_dataset(ds,filename):
    with open(filename,'w') as fout:
        json.dump(ds,fout)

def load_dataset(filename):
    with open(filename) as fin:
        ds = json.load(fin)
    return ds

def load_align_dataset():
    return load_dataset('../align_ds.json')
def load_phrase_dataset():
    return load_dataset('../phrase_ds.json')
    

def make_datasets(save = False):
    cgn_ids = [f.split('/')[-1] for f in glob.glob(cgn_align +'fn*')]
    phrase_ds = []
    align_ds = []
    for i,cgn_id in enumerate(cgn_ids):
        print(cgn_id,i,len(cgn_id))
        align = Align(cgn_id)
        phrase_ds.extend( phrases_to_dataset(align.phrases) )
        align_ds.append(align_to_dataset_line(align) )
    if save:
        save_dataset(phrase_ds,'../phrase_ds.json')
        save_dataset(align_ds,'../align_ds.json')
    return phrase_ds, align_ds
        



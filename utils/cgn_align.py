# part of the ASTA project to align wav2vec and cgn with nw

import glob
import random
import os
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
        Align(cgn_id)
        
    

    
class Align:
    '''align object to align cgn textgrid and wav2vec transcript.
    '''
    def __init__(self,cgn_id):
        self.cgn_id = cgn_id
        self.textgrid = Textgrid.objects.get(cgn_id = cgn_id)
        self.awd_words = list(self.textgrid.word_set.all())
        self.awd_text = ' '.join([w.awd_word for w in self.awd_words])
        self._set_wav2vec_table_and_text()
        self._set_align()
        self._set_phrases()

    def _set_wav2vec_table_and_text(self):
        self.wav2vec_base_filename = cgn_wav2vec_dir + self.textgrid.cgn_id
        self.wav2vec_table=load_table(self.wav2vec_base_filename+'.table')
        self.wav2vec_text=load_text(self.wav2vec_base_filename+'.txt')

    def _set_align(self):
        self.align_filename = cgn_align + self.cgn_id
        if os.path.isfile(self.align_filename): 
            with open(self.align_filename) as fin:
                self.align = fin.read()
        else:
            self.align = nw.nw(self.awd_text, self.wav2vec_text)
            with open(self.align_filename, 'w') as fout:
                fout.write(self.align)
        

    def _set_phrases(self):
        phrases = self.textgrid.phrases()
        o = align_phrases_with_aligned_text(phrases,self.aligned_cgn_text,
            self.awd_words)
        self.phrases = []
        for phrase,p, start_index, end_index in o:
            p = Phrase(phrase,p,self,start_index, end_index)
            self.phrases.append(p)
            

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
            i += 1
            o.append(self.wav2vec_table[i])
        return o

class Phrase:
    def __init__(self,phrase,p=None,align=None,start_index=None,end_index=None):
        self.phrase = phrase
        self.p = p
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
        m += 'alignment'.ljust(12) + self.alignment()
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
        start_ok= equal_with_delta(self.start_time, self.wav2vec_start_time, d)
        end_ok = equal_with_delta( self.end_time, self.wav2vec_end_time, d)
        if start_ok and end_ok: return 'good'
        if start_ok:  return 'start match'
        if end_ok:  return 'end match'
        if text_ok: return 'middle match'
        return 'bad'

    

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
        last_word_indices = word_indices[:]
        word_indices = _find_word_indices(phrase, word_list)
        between_word_indices = _find_between_word_indices(last_word_indices,
            word_indices)
        if between_word_indices:
            bt = _make_between_phrase(between_word_indices,word_list)
            start = _find_end_index(bt,text,start, start)
        if not _consecutive_indices(word_indices):
            p = _add_missing_words(phrase,word_list)
        else: p = phrase[:]
        pt = phrase_to_text(p)
        end = _find_end_index(pt,text,start, start)
        if not end:
            output.append([phrase,None,None,None])
        else:
            output.append([phrase,p,start,end])
            start = end
    return output

def _find_word_indices(phrase, word_list):
    indices = []
    for word in phrase:
        indices.append(word_list.index(word))
    return indices

def _find_between_word_indices(last_word_indices, word_indices):
    if last_word_indices == []: 
        if word_indices[0] == 0: return None
        n = word_indices[0] 
        offset = 0
    elif word_indices[0] - 1 == last_word_indices[-1]: return None
    else: 
        n = word_indices[0] - last_word_indices[-1] -1
        offset = last_word_indices[-1] +1
    return list(range(offset, n + offset))

def _make_between_phrase(indices,words):
    phrase = [words[i] for i in indices]
    text = phrase_to_text(phrase)
    return text

def _consecutive_indices(indices):
    for i,index in enumerate(indices):
        if not i < len(indices) -1:break
        if not index == indices[i+1] -1: return False
    return True

def _add_missing_words(word_indices,word_list):
    start = word_indices[0]
    end = word_indices[-1] + 1
    return word_list[start:end]


    
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
    

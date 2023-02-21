import glob

cgn_wav2vec_dir = '/vol/tensusers3/mbentum/cgn_wav2vec/'

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

# continued in cgn_audio_extract with all cgn textgrids in database
# erepo
    

        
    

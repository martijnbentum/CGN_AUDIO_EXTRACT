import glob
import locations
import os

filename_dictionary = None

def make_filename_list():
    fn = glob.glob(locations.cgn_awd_dir +'**', recursive=True)
    o = [x.split('comp-')[-1] for x in fn]
    save_filename_list(filename_list)
    return o

def save_filename_list(filename_list = None):
    if not filename_list: filename_list = make_filename_list()
    with open('../filename_list','w') as fout:
        fout.write('\n'.join(filename_list))

def load_filename_list():
    f = '../filename_list'
    if not os.path.isfile(f): return False
    with open(f,'r') as fin:
        filename_list = fin.read().split('\n')
    return filename_list
    

def make_filename_to_component_dictionary(filename_list = None):
    if not filename_list: 
        filename_list = load_filename_list()
        if not filename_list: filename_list = make_filename_list()
    l = filename_list
    return {x.split('/')[-1].split('.')[0]:x.split('/')[0] for x in l}
    
def filename_to_component(filename):
    global filename_dictionary
    if filename_dictionary == None: 
        d = make_filename_to_component_dictionary()
        filename_dictionary = d
    stem = filename.split('.')[0]
    try: return filename_dictionary[stem]
    except KeyError:
        print(filename, 'is might not be part of cgn')
        raise ValueError(stem + 'not in filename dictionary')

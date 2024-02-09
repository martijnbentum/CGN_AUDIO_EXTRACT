from . import filename_to_component
import glob
from . import locations
import os


def isfile(filename):
    if not os.path.isfile(filename): 
        raise FileNotFoundError('file: ' + filename + ' does not exist')

def isdir(dir_name):
    if not os.path.isdir(dir_name): 
        raise FileNotFoundError('directory: ' + dir_name + ' does not exist')

def unzip_file(filename):
    isfile(filename)
    os.system('gunzip ' + filename)

def make_writable(filename):
    isfile(filename)
    if cgn_original(filename): 
        raise ValueError('will not change original cgn files')
    os.system('chmod +w ' + filename)
    
def to_utf8(filename):
    isfile(filename)
    make_writable(filename)
    os.system('iconv -f iso-8859-1 -t utf-8 ' + filename + ' -o temp')
    os.system('mv temp ' + filename)

def copy_file_to_local(filename, goal_dir):
    isfile(filename)
    isdir(goal_dir)
    os.system('cp ' + filename + ' ' + goal_dir)

def cgn_original(filename):
    return locations.cgn_dir in filename 

def exists_locally(filename, local_dir):
    if not filename.startswith('fn'): filename = filename.split('/')[-1]
    if filename.endswith('.gz'): filename = filename.rstrip('.gz')
    filename = local_dir + filename
    if os.path.isfile(filename): return filename
    fn = glob.glob(filename+'*')
    if fn: return fn[0]
    return False

def make_cgn_path_filename(filename, directory = locations.cgn_awd_dir):
    component = filename_to_component.filename_to_component(filename)
    f = directory + 'comp-' + component.name
    if 'fn' in filename: f += '/nl/' + filename
    elif 'fv' in filename: f += '/vl/' + filename
    else: raise ValueError('filename not recognized')
    if os.path.isfile(f): return f
    fn = glob.glob(f + '*')
    if fn: return fn[0]
    raise ValueError(f,fn, 'could not find a file in the cgn corpus')

def handle_original_cgn_awd_file(filename, force = False):
    output_filename = exists_locally(filename, locations.local_awd)
    if output_filename and not force: return output_filename
    filename = make_cgn_path_filename(filename)
    copy_file_to_local(filename, locations.local_awd)
    filename = locations.local_awd + filename.split('/')[-1]
    if filename.endswith('.gz'): 
        unzip_file(filename)
        filename = filename.rstrip('.gz')
    to_utf8(filename)
    return filename

def handle_original_cgn_fon_file(filename, force = False):
    output_filename = exists_locally(filename, locations.local_fon)
    if output_filename and not force: return output_filename
    filename = make_cgn_path_filename(filename, locations.cgn_fon_dir)
    copy_file_to_local(filename, locations.local_fon)
    filename = locations.local_fon + filename.split('/')[-1]
    if filename.endswith('.gz'): 
        unzip_file(filename)
        filename = filename.rstrip('.gz')
    to_utf8(filename)
    return filename


def handle_original_cgn_ort_file(filename, force = False):
    print(filename)
    output_filename = exists_locally(filename, locations.local_ort)
    if output_filename and not force: return output_filename
    filename = make_cgn_path_filename(filename, locations.cgn_ort_dir)
    copy_file_to_local(filename, locations.local_ort)
    filename = locations.local_ort + filename.split('/')[-1]
    if filename.endswith('.gz'): 
        unzip_file(filename)
        filename = filename.rstrip('.gz')
    to_utf8(filename)
    print(filename)
    return filename


def _handle_language(language):
    if language == 'dutch': language = 'fn'
    elif language == 'netherlandic': language = 'fn'
    elif language == 'netherlandic dutch': language = 'fn'
    elif language == 'flemish': language = 'fv'
    elif language == 'belgian dutch': language = 'fv'
    elif language == 'flemish dutch': language = 'fv'
    if language not in ['fn','fv']: raise ValueError('language not recognized')
    return language

def _get_transcription_type_function(transcription):
    if transcription == 'awd':
        function = handle_original_cgn_awd_file
    elif transcription == 'fon':
        function = handle_original_cgn_fon_file
    elif transcription == 'ort':
        function = handle_original_cgn_ort_file
    else: raise ValueError('transcription not recognized')
    return function

def _handle_files(transcription = 'awd',language = 'fn'):
    error = []
    language = _handle_language(language)
    function = _get_transcription_type_function(transcription)
    for key in filename_to_component.make_filename_to_component_dictionary():
        if key.startswith(language):
            print(key)
            try: function(key)
            except:error.append(key)
    return error

def handle_all_dutch_awd_files():
    return _handle_files(transcription = 'awd',language = 'fn')

def handle_all_dutch_fon_files():
    return _handle_files(transcription = 'fon',language = 'fn')

def handle_all_dutch_ort_files():
    return _handle_files(transcription = 'ort',language = 'fn')

def handle_all_flemish_awd_files():
    return _handle_files(transcription = 'awd',language = 'fv')

def handle_all_flemish_fon_files():
    return _handle_files(transcription = 'fon',language = 'fv')

def handle_all_flemish_ort_files():
    return _handle_files(transcription = 'ort',language = 'fv')


def handle_all_dutch_files():
    errors = []
    errors += handle_all_dutch_awd_files()
    errors += handle_all_dutch_fon_files()
    errors += handle_all_dutch_ort_files()
    return errors

def handle_all_flemish_files():
    errors = []
    errors += handle_all_flemish_awd_files()
    errors += handle_all_flemish_fon_files()
    errors += handle_all_flemish_ort_files()
    return errors

    

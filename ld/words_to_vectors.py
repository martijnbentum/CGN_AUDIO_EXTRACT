'''
Module to convert words in audio files to vectors using a pretrained 
wav2vec model.

Used to create a dataset for training a model to predict stress in
MALD words.

'''

import glob
from utils import locations
from utils import to_vectors
import pickle

input_directory = locations.mald_variable_stress_wav
output_directory = locations.mald_variable_stress_pretrain_vectors

occlusion_input_dir= locations.mald_variable_stress_occlusions_wav
occlusion_output_dir=locations.mald_variable_stress_occlusions_pretrain_vectors

def get_name(filename):
    '''remove directory and extension from filename'''
    return filename.split('/')[-1].split('.')[0]

def all_audio_files_to_pretrain_vectors(input_directory = input_directory, 
    output_directory = output_directory, processor = None, model = None,
    end_filename = ''):
    '''uses pretrained model to convert audio files to vectors'''
    if not processor:
        processor, model = to_vectors.load_pretrained_model()
    files = glob.glob(input_directory + '*' + end_filename + '.wav')
    for f in files:
        audio_file_to_vector(f, processor, model, output_directory)

def all_mald_audio_files_to_pretrain_vectors(processor = None, model = None):
    if not processor:
        processor, model = to_vectors.load_pretrained_model()
    input_directory = locations.mald_word_recordings
    output_directory = locations.mald_pretrain_vectors
    all_audio_files_to_pretrain_vectors(input_directory, output_directory,
        processor, model)
    

def audio_file_to_vector(filename, processor, model, output_directory = ''):
    '''converts audio file to vectors and saves them to a pickle file'''
    print('handling file: ' + filename)
    name = get_name(filename)
    output = to_vectors.audio_to_pretrained_outputs(
        filename,processor = processor, model = model)
    pickle.dump(output, open(output_directory + name + '.pickle', 'wb'))
    return output

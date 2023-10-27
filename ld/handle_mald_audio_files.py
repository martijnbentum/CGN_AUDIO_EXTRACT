'''
Module to convert words in audio files to vectors using a pretrained 
wav2vec model.

Used to create a dataset for training a model to predict stress in
MALD words.

'''

import glob
from utils import to_vectors


input_directory = '/vol/tensusers/mbentum/variable_stress_syllable/wav_16khz/'
output_directory = '/vol/tensusers/mbentum/variable_stress_syllable/'
output_directory += 'pretrain_wav2vec_vectors/'

def get_name(filename):
    '''remove directory and extension from filename'''
    return filename.split('/')[-1].split('.')[0]

def handle_audio_files(input_directory = input_directory, 
    output_directory = output_directory, processor = None, model = None):
    '''uses pretrained model to convert audio files to vectors'''
    if not processor:
        processor, model = to_vectors.load_pretrained_model()
    files = glob.glob(directory + '*.wav')
    for f in files:
        handle_audio_file(f, processor, model)

def handle_audio_file(filename, processor, model, output_directory = ''):
    '''converts audio file to vectors and saves them to a pickle file'''
    print('handling file: ' + filename)
    name = get_filename(filename)
    output = to_vectors.audio_to_pretrained_outputs(
        filename,processor = processor, model = model)
    pickle.dump(output, open(output_directory + name + '.pickle', 'wb'))
    return output

from utils import audio
from utils import locations
from utils import to_vectors
import numpy as np
from transformers import Wav2Vec2ForPreTraining as pt
from transformers import Wav2Vec2Processor 
from transformers import Wav2Vec2Model

repo_dir = '/vol/tensusers/mbentum/CGN_AUDIO_EXTRACT/repo/'
default_checkpoint = repo_dir + 'o_first_test/checkpoint-6300'

pretrained_small="facebook/wav2vec2-xls-r-300m"
pretrained_big ="facebook/wav2vec2-xls-r-2b"

def audio_file_to_codebook_indices(audio_filename, processor, model, model_pt,
    start = 0.0, end = None, save = True, verbose = True):
    if verbose: print('handling', audio_filename)
    inputs = audio_to_inputs(audio_filename, processor, start, end)
    outputs = inputs_to_outputs(inputs, model)
    cv = outputs_to_codevectors(outputs, model_pt)
    codebook = load_codebook(model_pt)
    ci = codevectors_to_codebook_indices(cv, codebook)
    if save: 
        filename = audio_filename_to_codebook_indices_filename(audio_filename)
        print('saving',filename)
        save_codebook_indices(ci, filename)
    return ci

def load_pretrained_processor_model(version = 'small'):
    if version == 'small': name = pretrained_small
    else: name = pretrained_big
    # loading a non specific processor do not know whether this is ok
    processor = Wav2Vec2Processor.from_pretrained(default_checkpoint)
    model = Wav2Vec2Model.from_pretrained(name,
        cache_dir = locations.cache_dir)
    model_pt = pt.from_pretrained(pretrained_small)
    return processor, model, model_pt

def audio_to_array(audio_filename, start, end):
    return audio.load_audio(audio_filename,start, end)

def array_to_inputs(array, processor):
    inputs = processor(array, sampling_rate=16_000, return_tensors='pt',
        padding= True)
    return inputs

def audio_to_inputs(audio_filename, processor, start = 0.0, end = None):
    array = audio_to_array(audio_filename,start, end)
    inputs = array_to_inputs(array, processor)
    return inputs

def inputs_to_outputs(inputs, model):
    outputs = model(inputs.input_values, output_hidden_states=True)
    return outputs

def outputs_to_codevectors(outputs, model_pt):
    code_vectors, tensor = model_pt.quantizer(outputs.extract_features)
    return code_vectors.detach().numpy()[0]

def load_codebook(model_pt):
    codebook = model_pt.quantizer.codevectors
    return codebook.detach().numpy()[0]

def codevector_to_codebook_indices(codevector, codebook):
    slice_index = codebook.shape[-1]
    q1, q2 = codevector[:slice_index], codevector[slice_index:]
    index1 = get_row_index_of_vector_in_matrix(q1, codebook)
    index2 = get_row_index_of_vector_in_matrix(q2, codebook)
    codebook_indices = (index1, index2)
    return codebook_indices

def codevectors_to_codebook_indices(codevectors, codebook):
    codebook_indices = []
    for codevector in codevectors:
        ci = codevector_to_codebook_indices(codevector, codebook)
        codebook_indices.append(ci)
    return codebook_indices

def get_row_index_of_vector_in_matrix(vector, matrix):
    return np.argwhere((vector == matrix).all(1)).flatten()[0]


def audio_filename_to_codebook_indices_filename(audio_filename):
    f = audio_filename.split('audio/wav/')[-1].split('.wav')[0]
    f = f.replace('/', '_')
    f = locations.codebook_indices_dir + f + '.npy'
    return f

def save_codebook_indices(codebook_indices, filename):
    np.save(filename, codebook_indices)

def save_codebook(codebook):
    f = locations.codebook_indices_dir + 'codebook.npy'
    np.save(f, codebook)

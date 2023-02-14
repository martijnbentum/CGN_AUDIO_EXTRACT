from utils import hidden_state
from utils import phonemes
import numpy as np

def get_vocabs():
    '''get the vocabs to get the indices of the phonemes 
    to translate the indices in the prob matrix to phonemes
    '''
    vocab = hidden_state.load_vocab()
    ipa_vocab = phonemes.convert_simple_sampa_vocab_to_ipa_symbols(vocab)
    ipa_reverse_vocab = hidden_state.reverse_vocab(ipa_vocab)
    return vocab, ipa_vocab, ipa_reverse_vocab

vocab, ipa_vocab, ipa_reverse_vocab = get_vocabs()

def matrix_to_phoneme_labels(matrix):
    '''get the winning phoneme label for each column in the prob matrix.'''
    indices = np.argmax(matrix,1)
    phoneme_labels = []
    for index in indices:
        phoneme_labels.append(ipa_reverse_vocab[index])
    return phoneme_labels
        
def synthetic_probability_ditribution_to_indices(spd):
    '''find the correct index (in the prob matrix) for each phoneme
    in the synthetic prob distribution (based on the bpc)
    '''
    indices = []
    for phoneme in spd.keys():
        indices.append(ipa_vocab[phoneme])
    return indices

def normalize_prob_vector(v, epsilon = 0.00001):
    '''normalize a prob vector
    the mlp prob output for the phonemes label for a given frame
    need to be normalized after removing the winning (and other phonemes)
    from the mlp output
    v   prob vector from the prob matrix corresponding to a given frame
    epsilon     small value to prevend 0 errors in the kl computation
    '''
    v = v+ epsilon
    return v / np.sum(v)

def compute_mlp_output_and_synthetic_bpc(matrix):
    '''compute the kl divergence for each fram in a prob matrix
    the output from a mlp classifier for a section of audio.
    '''
    bpcs = phonemes.make_bpcs()
    phoneme_labels = matrix_to_phoneme_labels(matrix)
    output = []
    for i,phoneme in enumerate(phoneme_labels):
        try:bpc = bpcs.find_bpc(phoneme)
        except: 
            output.append([phoneme,i,None])
            continue
        spd = bpc.synthetic_probability_ditribution(phoneme)
        indices = synthetic_probability_ditribution_to_indices(spd)
        p = normalize_prob_vector(matrix[i,indices])
        q = list(spd.values())
        output.append([phoneme,i,kl_divergence(p,q)])
    return output

def kl_divergence(p,q):
    '''compute the kullback leibler divergence
    p   observed probs
    q   model probs
    '''
    return np.sum(p*np.log(p/q))
        

    

    

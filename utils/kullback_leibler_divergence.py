from utils import hidden_state
from utils import phonemes
import numpy as np

def get_vocabs():
    vocab = hidden_state.load_vocab()
    ipa_vocab = phonemes.convert_simple_sampa_vocab_to_ipa_symbols(vocab)
    ipa_reverse_vocab = hidden_state.reverse_vocab(ipa_vocab)
    return vocab, ipa_vocab, ipa_reverse_vocab

vocab, ipa_vocab, ipa_reverse_vocab = get_vocabs()

def matrix_to_phoneme_labels(matrix):
    indices = np.argmax(matrix,1)
    phoneme_labels = []
    for index in indices:
        phoneme_labels.append(ipa_reverse_vocab[index])
    return phoneme_labels
        
def synthetic_probability_ditribution_to_indices(spd):
    indices = []
    for phoneme in spd.keys():
        indices.append(ipa_vocab[phoneme])
    return indices

def normalize_prob_vector(v, epsilon = 0.00001):
    v = v+ epsilon
    return v / np.sum(v)

def prepare_mlp_output_and_synthetic_bpc(matrix):
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
    # print('p',p)
    # print('q',q)
    return np.sum(p*np.log(p/q))
        

    

    

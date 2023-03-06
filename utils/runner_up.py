import pickle
from utils import kullback_leibler_divergence
from utils import phonemes
import numpy as np
from matplotlib import pyplot as plt
   
def _prepare_runner_up_dict(frame,d):
    if not d: d = {'pretrained':{},'ctc':{}}
    if not frame.kl: return None, None, d
    mt = 'pretrained' if not frame.ctc else 'ctc'
    l = frame.layer
    if not l in d[mt].keys(): d[mt][l] = {}
    return mt, l, d

def count_runner_up(frame, d = None):
    model_type, layer, d = _prepare_runner_up_dict(frame,d)
    if not model_type: return d
    fd = d[model_type][layer]
    phoneme = frame.phoneme
    if phoneme not in fd.keys(): fd[phoneme] = {}
    runner_up = frame.phoneme_probability_vector[0][0]
    if runner_up not in fd[phoneme].keys(): fd[phoneme][runner_up] = 1
    else: fd[phoneme][runner_up] += 1
    bpc_name = frame.bpc.name
    if bpc_name not in fd.keys(): fd[bpc_name] ={'total':0,'part_of':0}
    fd[bpc_name]['total'] +=1
    if frame.bpc.part_of(runner_up): 
        fd[bpc_name]['part_of'] +=1
    return d

def count_runner_ups(frames, d = None):
    if not d: d = {}
    for frame in frames:
        d = count_runner_up(frame, d)
    return d

def cgn_ids_to_runner_up_counts(cgn_ids, filename = ''):
    d = None
    for cgn_id in cgn_ids:
        kla = kullback_leibler_divergence.load_kl_audio(cgn_id)
        frames = kla.klframes_ctc + kla.klframes_pretrained
        d = count_runner_ups(frames, d)
    if filename:
        with open(filename, 'wb') as fout:
            pickle.dump(d, fout)
    return d

def phoneme_runner_up(phoneme, runner_ups, bpcs = None):
    if not bpcs: bpcs = phonemes.make_bpcs()
    bpc = bpcs.find_bpc(phoneme)
    in_bpc, out_bpc = 0, 0
    for k, v in runner_ups.items():
        if bpc.part_of(k): in_bpc += v
        else: out_bpc += v
    total = in_bpc + out_bpc
    perc = round(in_bpc / total * 100,2)
    return perc, in_bpc, out_bpc, total

def get_phonemes(bpcs = None):
    if not bpcs: bpcs = phonemes.make_bpcs()
    p = []
    for bpc in bpcs.bpcs.values():
        p.extend( list(bpc.bpc_set) )
    return p

def plot_matrix(m, row_labels, column_ticks, column_labels):
    plt.figure()
    plt.rc('ytick',labelsize=24)
    plt.rc('xtick',labelsize=24)
    plt.imshow(m) 
    plt.colorbar()
    plt.xticks(column_ticks, column_labels)
    plt.yticks(ticks = list(range(len(row_labels))), labels=row_labels)
    

def runner_up_matrix(runner_up_dict , model_type = 'pretrained'):
    d = runner_up_dict[model_type]
    layers = [1,3,6,9,12,18,21,24]
    if 'cnn_features' in d.keys(): layers = ['cnn_features'] + layers
    p = get_phonemes()
    ncolumns = len(layers)
    nrows = len(p)
    values = nrows * ncolumns
    m = np.zeros((1,values)).reshape(nrows,ncolumns)
    for i, phoneme in enumerate(p):
        for j, layer in enumerate(layers):
            perc, _, _, _ = phoneme_runner_up(phoneme,d[layer][phoneme])
            m[i,j] = perc
    return p, layers, ncolumns, nrows, m


def plot_runner_up_matrix(runner_up_dict = None):
    if not runner_up_dict: 
        runner_up_dict = pickle.load(open('../runner_up_150.pickle','rb')) 
    d = runner_up_dict
    p,_, _, _, pretrained_matrix = runner_up_matrix(d)
    _,_, _, _, ctc_matrix = runner_up_matrix(d, model_type = 'ctc')
    fig, (ax1,ax2) = plt.subplots(1,2)
    plt.rcParams.update({'font.size':20})
    plt.rc('ytick',labelsize=24)
    plt.rc('xtick',labelsize=20)
    ax1.imshow(pretrained_matrix)
    ax1.set_xticks([0,4,8],['cnn',9,24])
    ax1.set_yticks(ticks = list(range(len(p))), labels=p)
    ax1.set_title('pretrained')
    plt.imshow(ctc_matrix)
    ax2.set_xticks([0,3,7],[1,9,24])
    ax2.set_yticks(ticks = list(range(len(p))), labels=p)
    ax2.set_title('ctc')
    plt.colorbar()
    plt.tight_layout()
    
    
    
    
    
    
    

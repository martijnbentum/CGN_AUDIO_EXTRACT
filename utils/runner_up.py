
   
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
        kla = load_kl_audio(cgn_id)
        frames = kla.klframes_ctc + kla.klframes_pretrained
        d = count_runner_ups(frames, d)
    if filename:
        with open(filename, 'wb') as fout:
            pickle.dump(d, fout)
    return d

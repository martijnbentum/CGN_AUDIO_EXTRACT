import os
from utils import filename_to_component

cgn_ids_fn = '../cgn_ids'
speaker_ids_fn = '../speaker_ids'

def _make_cgn_ids():
    from text.models import Audio
    a = Audio.objects.all()
    cgn_ids = [x.cgn_id for x in Audio.objects.all()]
    with open(cgn_ids_fn,'w') as fout:
        fout.write('\n'.join(cgn_ids))
    return cgn_ids

def load_cgn_ids():
    if not os.path.isfile(cgn_ids_fn): return _make_cgn_ids()
    with open(cgn_ids_fn) as fin:
        cgn_ids = fin.read().split('\n')
    return cgn_ids
    
def _make_speaker_ids():
    from text.models import Speaker
    s = Speaker.objects.all()
    speaker_ids = [x.speaker_id for x in Speaker.objects.all()]
    with open(speaker_ids_fn,'w') as fout:
        fout.write('\n'.join(speaker_ids))
    return speaker_ids

def load_speaker_ids():
    if not os.path.isfile(speaker_ids_fn): return _make_speaker_ids()
    with open(speaker_ids_fn) as fin:
        speaker_ids = fin.read().split('\n')
    return speaker_ids

def cgn_id_to_component(cgn_id):    
    return filename_to_component.filename_to_component(cgn_id)

def cgn_id_to_audio(cgn_id):
    from text.models import Audio
    return Audio.objects.get(cgn_id = cgn_id)

def speaker_id_to_speaker(speaker_id):
    from text.models import Speaker
    return Speaker.objects.get(speaker_id = speaker_id)

def cgn_id_to_speakers(cgn_id):
    from text.models import Textgrid, Speaker 

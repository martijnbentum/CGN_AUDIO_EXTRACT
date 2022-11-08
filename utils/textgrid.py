from utils import handle_file
from utils import identifiers
from utils import locations
import textgrids

def get_awd_textgrid_filename(filename):
    '''
    filename can be cgn_id e.g. fn00088
    '''
    f = handle_file.exists_locally(filename,locations.local_awd)
    if not f: f= handle_file.handle_original_cgn_awd_file(filename)
    return f

def get_fon_textgrid_filename(filename):
    '''
    filename can be cgn_id e.g. fn00088
    '''
    f = handle_file.exists_locally(filename,locations.local_fon)
    # if not f: f= handle_file.handle_original_cgn_fon_file(filename)
    if not f: return ''
    return f

def get_ort_textgrid_filename(filename):
    '''
    filename can be cgn_id e.g. fn00088
    '''
    f = handle_file.exists_locally(filename,locations.local_ort)
    # if not f: f= handle_file.handle_original_cgn_fon_file(filename)
    if not f: return ''
    return f
    
def load_awd_textgrid(filename):
    f = get_awd_textgrid_filename(filename)
    return textgrids.TextGrid(f)

def textgrid_to_speakers(textgrid):
    from text.models import Speaker
    speaker_ids = []
    for tier_name in textgrid:
        if '_' not in tier_name: 
            try:speaker_ids.append(identifiers.speaker_id_to_speaker(tier_name))
            except Speaker.DoesNotExist:
                print(tier_name,'is not a speaker')
    return speaker_ids

def load_in_awd_textgrid(filename):
    from text.models import Audio,Component,Textgrid
    filename = get_awd_textgrid_filename(filename)
    cgn_id = filename.split('/')[-1].split('.')[0]
    try: return Textgrid.objects.get(cgn_id = cgn_id)
    except Textgrid.DoesNotExist: pass
    d ={}
    d['awd_filename'] = filename
    d['cgn_id'] = cgn_id
    d['audio'] = identifiers.cgn_id_to_audio(cgn_id)
    textgrid = load_awd_textgrid(filename)
    speakers = textgrid_to_speakers(textgrid)
    d['component'] = identifiers.cgn_id_to_component(filename)
    d['nspeakers'] = len(speakers)
    tg = Textgrid(**d)
    tg.save()
    for speaker in speakers:
        tg.speakers.add(speaker)
    return tg

def load_in_all_awd_textgrids():
    cgn_ids = identifiers.load_cgn_ids()
    tgs = []
    for cgn_id in cgn_ids:
        print(cgn_id)
        tgs.append(load_in_awd_textgrid(cgn_id))
    return tgs
    


def intervals_to_dict(intervals):
    d={}
    for i, iv in enumerate(intervals):
        d[i] = '\t'.join(map(str,[iv.text,iv.xmin,iv.xmax]))
    return d



import glob
from . import locations
from . import filename_to_component
import os
import subprocess
import librosa

def load_audio(filename, start = 0.0, end=None):
	if not end: duration = None
	else: duration = end - start
	audio, sr = librosa.load(filename, sr = 16000, offset=start, duration=duration)
	return audio

def load_recording(recording, start = 0.0,end = None):
	audio = load_audio(recording.wav_filename,start,end)
	return audio
	

def load_audio_section(start_time,end_time,filename,audio=None):
    sampling_rate = 16000
    if not audio: audio = load_audio(filename)
    return audio[int(start_time*sampling_rate):int(end_time*sampling_rate)]

def make_audio_filenames_list(restrict_to_dutch = True, save = True):
    fn = glob.glob(locations.cgn_audio_dir +'**', recursive=True)
    output = []
    for f in fn:
        if '/nl/fn' in f: output.append(f)
    if save:
        with open('../audio_filenames','w') as fout:
            fout.write('\n'.join(output))
    return output

def load_audio_filenames_list():
    f = '../audio_filenames'
    if not os.path.isfile(f): make_audio_filenames_list()
    with open(f) as fin:
        o = fin.read()
    return o.split('\n')
        
def sox_info(filename):
    o = subprocess.run(['sox','--i',filename],stdout=subprocess.PIPE)
    return o.stdout.decode('utf-8')

def clock_to_duration_in_seconds(t):
    hours, minutes, seconds = t.split(':')
    s = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    return s

def soxinfo_to_dict(soxinfo):
    x = soxinfo.split('\n')
    d = {}
    d['filename'] = x[1].split(': ')[-1].strip("'")
    d['nchannels'] = x[2].split(': ')[-1]
    d['sample_rate'] = x[3].split(': ')[-1]
    t = x[5].split(': ')[-1].split(' =')[0]
    d['duration'] = clock_to_duration_in_seconds(t)
    return d

def read_in_audio(filename, save = True):
    from text.models import Audio
    cgn_id = filename.split('/')[-1].split('.')[0]
    try: return Audio.objects.get(cgn_id = cgn_id)
    except Audio.DoesNotExist: pass
    sox = sox_info(filename)
    d = soxinfo_to_dict(sox)
    d['component']= filename_to_component.filename_to_component(cgn_id)
    d['cgn_id'] = cgn_id
    a = Audio(**d)
    a.save()
    return a
        
def read_in_all_audios(filename_list = None):
    if not filename_list: filename_list = load_audio_filenames_list()
    audios = []
    for filename in filename_list:
        print(filename)
        audio = read_in_audio(filename)
        audios.append(audio)
    return audios
        

    
def load_audio_fon_phrase(fon_phrase):
    p = fon_phrase
    return load_audio(p.textgrid.audio.filename,p.start_time, p.end_time)
    
        
        

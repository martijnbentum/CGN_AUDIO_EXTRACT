import glob
from . import locations
from . import filename_to_component
import os
import subprocess
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert,butter,filtfilt

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
        if not f.endswith('.wav'): continue
        if restrict_to_dutch and '/nl/fn' in f:
            output.append(f)
        else:
            output.append(f)
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

def hilbert_envelope(audio, sr = 16_000):
    analytic_signal = hilbert(audio)
    amplitude_envelope = np.abs(analytic_signal) **2 *20
    frequency =50 
    order = 3
    b, a = butter(order, frequency / (0.5 * sr), btype='low', analog=False)
    amplitude_envelope = filtfilt(b, a, amplitude_envelope)
    return amplitude_envelope

def speech_envelope(audio, sr = 16_000, envelope_sr = 100, swd = 2.5):
    step_size = sr / envelope_sr
    window_size = round(step_size * swd)
    window = np.hamming(window_size)
    pad_size = round((window_size - step_size)/2)
    print(pad_size)
    audio = np.concatenate((np.zeros(pad_size),audio,np.zeros(pad_size)))
    steps = int(len(audio) / step_size) 
    print(step_size,window_size,window.shape,steps)
    envelope = np.zeros(steps)
    for step in range(steps):
        start = int(step * step_size)
        end = start + window_size
        if end > len(audio): break
        envelope[step] = sum((audio[start:end] * window)**2)
    return envelope, envelope_sr 
    
def plot_audio(audio, start_time = 0, end_time = None, sr = 16_000,
    plot_envelope = False):
    start_index = int(start_time * sr)
    end_index = int(end_time * sr) if end_time else len(audio)
    s = audio[start_index:end_index]
    time = [x/sr + start_time for x in range(len(s))]
    se, envelope_sr = speech_envelope(s)
    se_time = [x/envelope_sr + start_time for x in range(len(se))]
    he = hilbert_envelope(s)
    plt.clf()
    plt.plot(time,s,color = 'black', alpha = .7,linewidth = .5)
    plt.plot(se_time,se,color = 'blue', alpha = .9,linewidth = 1)
    plt.plot(time,he,color = 'red', alpha = .9,linewidth = 1)
    # plt.xticks(locs, [round(x/sr,2) for x in locs])
    plt.xlabel('seconds')
    plt.ylabel('Amplitude')
    plt.grid(alpha = .3)
    plt.show()
        
        

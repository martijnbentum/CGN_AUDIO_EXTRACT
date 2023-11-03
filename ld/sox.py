import os
import subprocess

def extract_section(input_filename, output_filename, start_time, end_time):
    duration = end_time - start_time
    cmd = "sox " + input_filename + " " + output_filename  
    cmd += " trim " + str(start_time) + " " + str(duration)
    print(cmd)
    os.system(cmd)

def add_silence(input_filename, output_filename, silence_before, 
    silence_after):
    cmd = "sox " + input_filename + " " + output_filename 
    cmd += " pad " + str(silence_before) + " " + str(silence_after)
    print(cmd)
    os.system(cmd)

def occlude_other(input_filename, output_filename, start_time, end_time):
    d = audio_info(input_filename)
    temp_filename = output_filename.replace(".wav", "_temp.wav") 
    extract_section(input_filename, temp_filename, start_time, end_time)
    start_silence = start_time
    end_silence = d['duration'] - end_time
    add_silence(temp_filename, output_filename, start_silence, end_silence)
    os.remove(temp_filename)

    
def audio_info(filename):
    o = sox_info(filename)
    return soxinfo_to_dict(o)
    
def sox_info(filename):
    o = subprocess.run(['sox','--i',filename],stdout=subprocess.PIPE)
    return o.stdout.decode('utf-8')

def soxinfo_to_dict(soxinfo):
    x = soxinfo.split('\n')
    d = {}
    d['filename'] = x[1].split(': ')[-1].strip("'")
    d['nchannels'] = x[2].split(': ')[-1]
    d['sample_rate'] = x[3].split(': ')[-1]
    t = x[5].split(': ')[-1].split(' =')[0]
    d['duration'] = clock_to_duration_in_seconds(t)
    return d

def clock_to_duration_in_seconds(t):
    hours, minutes, seconds = t.split(':')
    s = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    return s

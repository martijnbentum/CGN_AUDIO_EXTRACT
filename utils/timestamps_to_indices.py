from utils import locations
from utils import identifiers

import glob
import numpy as np

timestamps_fn = glob.glob(locations.timestamps_dir + '*.txt')
component_to_cgn_ids_dict = identifiers.component_to_cgn_ids_dict()

def get_comp_timestamp_files(comp = 'o'):
    output = []
    for f in timestamps_fn:
        file_id = f.split('/')[-1].split('.')[0]
        if file_id in component_to_cgn_ids_dict[comp]: output.append(f)
    return output


def load_timestamp_file(filename):
    output = []
    with open(filename) as fin:
        t = [line.split('\t') for line in fin.read().split('\n')]
    for line in t:
        if len(line) < 3: 
            print([line],'skipping')
            continue
        line[1] = float(line[1])
        line[2] = float(line[2])
        output.append(line)
    return output

def make_index_to_char_dict(filename, frame_duration = 0.02):
    ts = load_timestamp_file(filename)
    d = {}
    for line in ts:
        char = line[0]
        duration = timestamp_line_to_duration(line)
        if duration > frame_duration: 
            n = int(round( duration / frame_duration, 0))
            end_time = line[2]
            for ntimes in range(n):
                index = timestamp_to_index(end_time - ntimes*frame_duration,
                    frame_duration = frame_duration)
                d[index] = char
        else:
            index = timestamp_line_to_index(line,
                frame_duration = frame_duration)
            d[index] = char
    return d

def cgn_id_to_timestamp_file(cgn_id):
    for f in timestamps_fn:
        if cgn_id in f: return load_timestamp_file(f)
    print('could not find:',cgn_id)

def cgn_id_to_index_dict(cgn_id, frame_duration = 0.02):
    for f in timestamps_fn:
        if cgn_id in f: return make_index_to_char_dict(f,frame_duration)
    print('could not find:',cgn_id)

def timestamp_line_to_duration(line):
    return round(line[2] - line[1],2)

def timestamp_to_index(timestamp, frame_duration = 0.02, offset = 0):
    '''computes the index of a frame based on timestamp and frame duration
    if an offset is provided this is subtracted from the timestamp before
    index computation. 
    the timestamp files show the timestamps relative to the whole file
    extracting the hidden state frames can be extracte on shorter stretches,
    offset provides a means of converting between the whole file and short
    stretches. 
    the offset should be a multiple of frame duration
    you can use time to frame aligned time to compute such a time.
    '''
    check_if_time_is_frame_aligned(offset, frame_duration)
    if offset > timestamp:
        raise ValueError('offset:',offset,'is larger than timestamp:',timestamp)
    index = (timestamp - offset) / frame_duration
    if abs(index - round(index,0)) > 0.000001:
        raise ValueError(index, 'is not an integer')
    return int(round(index,0))

def timestamp_line_to_index(line, frame_duration = 0.02, offset = 0):
    '''computes index based on a line from a timestamp file.
    if an offset is passed this is subtracted from the timestamp before
    computing the index
    '''
    return timestamp_to_index(line[2], frame_duration, offset)

def index_to_timestamp(index, frame_duration = 0.02, offset = 0):
    '''computes timestamp based on index and frame duration.'''
    return index * frame_duration + offset

def time_to_frame_aligned_time(time_in_seconds, frame_duration = 0.02, 
    nearest_lower = True):
    '''creates a timestamp that is a multiple of frame duration
    to the nearest lower frame aligned time
    '''
    if check_if_time_is_frame_aligned(time_in_seconds, fail = False): 
        return time_in_seconds
    time_in_seconds = round(time_in_seconds,3)
    diff = (int(time_in_seconds*1000)%int(frame_duration*1000))/1000
    t = time_in_seconds - diff
    if not nearest_lower: t += frame_duration
    t = round(t,2)
    if t < 0: t = 0
    check_if_time_is_frame_aligned(t)
    return t

def check_if_time_is_frame_aligned(time_in_seconds, frame_duration = 0.02,
    fail = True):
    '''checks whether time is a multiple of frame duration,
    this is necessary to compute indices of timestamps with an offset
    '''
    if time_in_seconds == 0: return True
    if round(time_in_seconds*1000,2) % int(frame_duration *1000) != 0:
        m= 'time in seconds: '+ str(time_in_seconds) + ' '
        m += 'is not multiple of frame_duration: ' + str(frame_duration)
        if not fail: return False
        raise ValueError(m)
    return True
    
def start_end_time_to_indices(start, end, frame_duration = 0.02):
    start = round(start + frame_duration, 2)
    end = round(end + frame_duration /2, 2)
    indices = []
    for x in np.arange(start, end , frame_duration):
        i = timestamp_to_index(x,offset=start),timestamp_to_index(x)
        indices.append(i)
    return indices

def phrase_to_indices(phrase):
    return start_end_time_to_indices(phrase.start_time, phrase.end_time)

def get_phrase_start_end_times(textgrid):
    o = []
    if textgrid.fon_filename:
        for phrase in textgrid.fon_phrases():
            start=time_to_frame_aligned_time(phrase.start_time)
            end=time_to_frame_aligned_time(phrase.end_time,nearest_lower=False)
            o.append([start,end])
    else:
        for phrase in textgrid.phrases():
            start = time_to_frame_aligned_time(phrase[0].start_time)
            end=time_to_frame_aligned_time(phrase[-1].end_time,
                nearest_lower=False)
            o.append([start,end])
    return o


    

    

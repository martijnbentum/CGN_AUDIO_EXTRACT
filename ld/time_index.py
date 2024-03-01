import math

def index_to_time(index, frame_duration = 0.02):
    start = round(index * frame_duration,3)
    end = round(start + frame_duration,3)
    return start, end

def start_time_to_index(time, frame_duration = 0.02, greedy = False):
    index = int(time / frame_duration)
    if greedy: return index
    start_time,end_time = index_to_time(index, frame_duration)
    if start_time < time: return index + 1
    return index

def end_time_to_index(time, frame_duration = 0.02, greedy = False):
    index = math.ceil(time / frame_duration) 
    if greedy: return index
    start_time,end_time = index_to_time(index, frame_duration)
    if start_time > time: return index - 1
    return index

def _find_nudge_index(start_index, end_index, start_time, end_time,
    frame_duration):
    '''find the index that is farthest from the time slice of a frame
    and nudge it to the closest index that is within the time slice of a frame
    '''
    check_start,check_end = index_slice_to_time_slice(start_index,
        end_index, frame_duration)
    start_diff = check_start - start_time
    end_diff = end_time - check_end
    if start_diff > end_diff:
        return start_index - 1, end_index
    return start_index, end_index + 1
    

def time_slice_to_index_slice(start_time,end_time, frame_duration = 0.02):
    '''find the index tuple that will slice those vectors that
    falls within the time slice
    if the time slice is shorter than the frame duration, then
    it will select the index that is closest to the time slice
    '''
    if start_time > end_time:
            raise ValueError("start",start_time,"end",end_time, "bad")
    start_index = start_time_to_index(start_time, frame_duration)
    end_index = end_time_to_index(end_time, frame_duration)
    nudged = False
    if start_index == end_index: 
        # if the start and end index is the same, then nudge the index
        # that is farthest from the time slice of a frame
        start_index, end_index = _find_nudge_index(start_index, end_index,
            start_time, end_time, frame_duration)
        nudged = True
    if start_index > end_index: 
        if start_index - end_index > 1: 
            raise ValueError("start",start_index,"end",end_index, "bad")
        start_index,end_index = end_index,start_index
    # check whether the start and end index is correct
    check_start,check_end = index_slice_to_time_slice(start_index,
        end_index, frame_duration)
    duration = end_time - start_time
    # if the duration of the time slice is shorter than the frame duration
    # the index slice will not be contained witin the time slice
    if duration > frame_duration * 1.5 and not nudged:
        # if duration is longer than frame duration
        # the check_start should be later or equal than the start time
        # and the check_end should be earlier or equal than the end time
        assert start_time <= check_start and end_time >= check_end
    return start_index, end_index

def index_slice_to_time_slice(start_index,end_index, frame_duration = 0.02):
    if start_index > end_index:
            raise ValueError("start",start_index,"end",end_index, "bad")
    start_time,end_time = index_to_time(start_index, frame_duration)
    end_time = index_to_time(end_index, frame_duration)[0]
    return start_time, end_time

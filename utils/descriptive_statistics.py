from text.models import Audio,Speaker
import numpy as np
from utils import phrase


def component_to_description_dict():
    output = {'a':"Spontaneous conversations",
        'b':"Interviews with teachers of Dutch",
        'c':"Spontaneous telephone dialogues",
        'd':"Spontaneous telephone dialogues",
        'e':"Simulated business negotiations",
        'f':"Interviews/discussions/debates",
        'g':"Discussions/debates/ meetings",
        'h':"Lessons recorded in the classroom",
        'i':"Live (sports) commentaries",
        'j':"Newsreports/reportages",
        'k':"News",
        'l':"Commentaries/columns/reviews",
        'm':"Ceremonious speeches/sermons",
        'n':"Lectures/seminars",
        'o':"Read speech"
        }
    return output

def component_to_audio_dict():
    output = {}
    audios= Audio.objects.all()
    for audio in audios:
        key = audio.component.name
        if key not in output: output[key] = []
        output[key].append(audio)
    return output

def component_to_audio_duration_dict(audio_dict = None):
    output = {}
    for component, audios in audio_dict.items():
        if component not in output: output[component] = []
        for audio in audios:
            output[component].append(audio.duration)
    return output


def component_to_speaker_dict(audio_dict = None):
    if not audio_dict: audio_dict = component_to_audio_dict()
    output = {}
    for component, audios in audio_dict.items():
        if component not in output: output[component] = []
        for audio in audios:
            for textgrid in audio.textgrid_set.all():
                for speaker in textgrid.speakers.all():
                    if speaker not in output[component]: 
                        output[component].append(speaker)
    return output

def component_to_nspeakers_per_audio(audio_dict = None):
    if not audio_dict: audio_dict = component_to_audio_dict()
    output = {}
    for component, audios in audio_dict.items():
        if component not in output: output[component] = []
        for audio in audios:
            for textgrid in audio.textgrid_set.all():
                output[component].append(textgrid.nspeakers)
    return output
    

def markdown_table_header():
    h='id | component | # audio files | dur total (h) | min dur (s) '
    h += '| median dur (s) '
    h += '| max dur (s) | # speakers | min # spks | median # spks '
    h += '| max # spks'
    l ='--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---'
    return [h,l]

def markdown_table_row(component,description,audios,duration,min_audio,
    median_audio,max_audio,nspeakers,min_nspeakers,
    median_nspeakers,max_nspeakers):
    return '|'.join([component,description,str(audios),str(duration),
        str(min_audio),
        str(median_audio),str(max_audio),str(nspeakers),str(min_nspeakers),
        str(median_nspeakers),str(max_nspeakers)])



def statistics_per_component(audio_dict = None, speaker_dict = None,
    nspeakers_per_audio_dict = None):
    if not audio_dict: audio_dict = component_to_audio_dict()
    if not speaker_dict: speaker_dict = component_to_speaker_dict(audio_dict)
    audio_duration_dict = component_to_audio_duration_dict(audio_dict)
    description_dict = component_to_description_dict()
    output = markdown_table_header()
    for component, audios in audio_dict.items():
        description = description_dict[component]
        duration = round(np.sum([x.duration for x in audios])/ 3600,2)
        min_audio= np.min(audio_duration_dict[component])
        max_audio= np.max(audio_duration_dict[component])
        median_audio= np.median(audio_duration_dict[component])
        nspeakers = len(speaker_dict[component])
        min_nspeakers = np.min(nspeakers_per_audio_dict[component])
        max_nspeakers = np.max(nspeakers_per_audio_dict[component])
        median_nspeakers = np.median(nspeakers_per_audio_dict[component])
        print(component, len(audios), duration, min_audio, median_audio, 
            max_audio, nspeakers, min_nspeakers,median_nspeakers,max_nspeakers)
        output.append(markdown_table_row(component,description,
            len(audios),duration,
            min_audio,median_audio,max_audio,nspeakers,min_nspeakers,
            median_nspeakers,max_nspeakers))
    return '\n'.join(output)
            
             
def seconds_time_to_string(seconds):
    hours = int(seconds/3600)
    minutes = int((seconds - hours*3600)/60)
    seconds = int(seconds - hours*3600 - minutes*60)
    return f'{hours}:{minutes}:{seconds}'

def analyze_gender(cgn = None):
    if not cgn: cgn = phrase.load_cgn()
    output = []
    for gender in ['male','female']:
        o = [v for k,v in cgn.items() if v['phrases'][0]['gender'] == gender]
        duration = sum([x['duration'] for x in o])
        print(gender.ljust(9),len(o), seconds_time_to_string(duration))
        output.append(o)
    return output

def analyze_phrases(cgn = None):
    if not cgn: cgn = phrase.load_cgn()
    speaker_duration,durations,phrases= [],[],[]
    for k,v in cgn.items():
        p = v['phrases']
        speaker_duration.append( v['duration'] )
        durations.extend([x['duration'] for x in p])
        phrases.extend(p)
    print('total phrases:',len(phrases))
    print('min phrase duration:',np.min(durations))
    print('mean phrase duration:',np.mean(durations))
    print('median phrase duration:',np.median(durations))
    print('min speaker duration:',np.min(speaker_duration))
    print('mean speaker duration:',np.mean(speaker_duration))
    return phrases, durations, speaker_duration



    

def get_audio_statistics():
    return audio_count

from utils import general
from utils import textgrid

def speaker_textgrid_to_words(tg, speaker):
    from text.models import Word
    t =  tg.load_awd()
    intervals = t[speaker.speaker_id]
    words = []
    if tg.nspeakers > 1: 
        other_speakers = [x for x in tg.speakers.all() if x != speaker]
        other_speakers_intervals=[t[x.speaker_id] for x in other_speakers]
    for i,interval in enumerate(intervals):
        if interval.text.strip('.?!') in ['',' ','_','ggg','xxx']: continue
        if tg.nspeakers > 1:
            osi =other_speakers_intervals
            overlap = check_overlap_other_speakers(interval,osi)
        else: overlap = False
        try: 
            words.append(Word.objects.get(textgrid = tg,speaker =speaker, 
                awd_word_tier_index = i))
            continue
        except Word.DoesNotExist: pass
        d = {}
        d['word'] = interval.text.strip('.?!')
        d['word_phoneme'] = t[speaker.speaker_id +'_FON'][i].text
        d['phonemes'] = str(get_phonemes(t,speaker,interval))
        d['special_word'] = '*' in interval.text
        d['eos'] = '.' in interval.text or '?' in interval.text
        d['overlap'] = overlap
        d['awd_word_tier_index'] = i
        d['start_time'] = interval.xmin
        d['end_time'] = interval.xmax
        d['textgrid'] = tg
        d['speaker'] = speaker
        w = Word(**d)
        w.save()
        words.append(w)
    return words

def get_phonemes(t,speaker,interval):
    s1, e1 = interval.xmin, interval.xmax
    segment_intervals = t[speaker.speaker_id+'_SEG']
    overlapping_intervals = []
    for phoneme in segment_intervals:
        if phoneme.text == '': continue
        s2, e2 = phoneme.xmin, phoneme.xmax
        if general.overlap(s1,e1,s2,e2,strict = True): 
            overlapping_intervals.append(phoneme)
    phoneme_dict = textgrid.intervals_to_dict(overlapping_intervals)
    return phoneme_dict

def check_overlap_other_speakers(interval,other_speakers_intervals):
    for other_intervals in other_speakers_intervals:
        if check_overlap_interval(interval,other_intervals):
            return True
    return False

def check_overlap_interval(interval,other_intervals):
    s1, e1 = interval.xmin, interval.xmax
    for other_interval in other_intervals:
        if other_interval.text.strip('.?!') in ['',' ','_','ggg','xxx']: continue
        s2, e2 = other_interval.xmin, other_interval.xmax
        if general.overlap(s1,e1,s2,e2): return True
    return False


 
def textgrid_to_words(tg):
    t = tg.load_awd()
    speakers = tg.speakers.all()
    words = []
    for speaker in speakers:
        words.extend( speaker_textgrid_to_words(tg, speaker))
    return words

def load_in_all_words_from_all_textgrids():
    from text.models import Textgrid
    tgs = Textgrid.objects.all()
    words = []
    for i,tg in enumerate(tgs):
        print(i,tg)
        words.extend( textgrid_to_words(tg) )
    return words
        

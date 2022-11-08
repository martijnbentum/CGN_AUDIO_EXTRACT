from utils import general
from utils import textgrid

def speaker_textgrid_to_fon_phrases(tg, speaker):
    from text.models import Fon_phrase
    t =  tg.load_fon()
    intervals = t[speaker.speaker_id]
    phrases= []
    if tg.nspeakers > 1: 
        other_speakers = [x for x in tg.speakers.all() if x != speaker]
        other_speakers_intervals=[t[x.speaker_id] for x in other_speakers]
    for i,interval in enumerate(intervals):
        if interval.text.strip('.?!') in ['',' ','_','ggg','xxx','[]','#']: 
            continue
        if tg.nspeakers > 1:
            osi =other_speakers_intervals
            overlap = check_overlap_other_speakers(interval,osi)
        else: overlap = False
        try: 
            phrases.append(Fon_phrase.objects.get(textgrid=tg,speaker =speaker, 
                fon_tier_index= i))
            continue
        except Fon_phrase.DoesNotExist: pass
        d = {}
        d['phrase'] = interval.text.strip('.?!')
        d['overlap'] = overlap
        d['fon_tier_index'] = i
        d['start_time'] = interval.xmin
        d['end_time'] = interval.xmax
        d['textgrid'] = tg
        d['speaker'] = speaker
        fp = Fon_phrase(**d)
        fp.save()
        phrases.append(fp)
    return phrases


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
 
def textgrid_to_fon_phrases(tg):
    speakers = tg.speakers.all()
    phrases = []
    for speaker in speakers:
        phrases.extend( speaker_textgrid_to_fon_phrases(tg, speaker))
    return phrases

def load_in_all_fon_phrases_from_all_textgrids():
    from text.models import Textgrid
    tgs = Textgrid.objects.exclude(fon_filename = '')
    phrases = []
    for i,tg in enumerate(tgs):
        print(i,tg)
        phrases.extend( textgrid_to_fon_phrases(tg) )
    return phrases



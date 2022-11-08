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
        d['awd_word'] = interval.text.strip('.?!')
        d['awd_word_phoneme'] = t[speaker.speaker_id +'_FON'][i].text
        d['awd_phonemes'] = str(get_phonemes(t,speaker,interval))
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

def words_to_duration(words):
    return sum([word.duration for word in words])

def _check_end_of_phrase(word,phrase,end_on_eos,maximum_duration):
    '''checks whether phrase should end based on word status and
    user specified criteria.
    word exclusion: overlap or special word (i.e. a * or ggg code in the word)
    user criterion: use end of sentence / use maximum phrase duration
    adds the word to the phrase if the phrase does not end 
    '''
    to_long, end = False, False
    if phrase: duration = word.end_time - phrase[0].start_time
    else: duration = 0
    if maximum_duration: to_long = duration > maximum_duration
    # ipa_missing = word.word_ipa_phoneme == ''
    exclude_word =word.overlap or word.special_word #or ipa_missing
    if exclude_word: end = True
    elif to_long: end = True
    else: phrase.append(word)
    if end_on_eos and word.eos: end = True
    return end, duration, to_long, exclude_word

def _add_phrase_to_phrases(phrase,phrases,minimum_duration):
    '''whether to add the phrase to the phrases list based on minimum duration.
    '''
    if not phrase: return
    phrase_duration = words_to_duration(phrase)
    if minimum_duration: 
        if phrase_duration >= minimum_duration:
            phrases.append(phrase)
    else: phrases.append(phrase)

def words_to_phrases(words, end_on_eos = True, 
    minimum_duration = None, maximum_duration = None):
    '''split word list into phrases based on several criteria
    see _check_end_of_phrase
    end_on_eos          whether to end a phrase because of an end of sentence
                        token
    minimum_duration    whether to include a phrase (exclude if it is shorter
                        than minimum_duration if none will not be used
    maximum_duration    whether to end a phrase because it is longer than
                        maximum_duration if none will not be used
    '''
    phrases, phrase = [], []
    for i,word in enumerate(words):
        end,duration,to_long, exclude_word = _check_end_of_phrase(
            word,phrase,end_on_eos,maximum_duration)
        if i == len(words) -1: end = True
        if end:
            _add_phrase_to_phrases(phrase,phrases,minimum_duration)
            phrase = []
            if to_long and not exclude_word: 
                phrase.append(word)
                duration = word.duration
            to_long = False
    return phrases
        

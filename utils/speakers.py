from . import locations

gender = {'sex1':'male','sex2':'female','sexX':'unknown'}

def open_file():
    t = open(locations.cgn_speaker_file).read().split('\n')
    header = t[0].split('\t')
    data = [x.split('\t') for x in t[1:] if x]
    return header, data

def make_speaker_dict(line):
    d = {}
    d['information'] = '\t'.join(line)
    d['speaker_id'] = line[4]
    d['gender'] = gender[line[5]]
    try:birth_year = int(line[6])
    except ValueError: birth_year = None 
    d['birth_year'] = birth_year
    if birth_year: d['age'] = 2000 - birth_year
    else: d['age'] = None
    return d

def add_speaker(line):
    from text.models import Speaker
    d = make_speaker_dict(line)
    try: return Speaker.objects.get(speaker_id= d['speaker_id'])
    except Speaker.DoesNotExist: pass
    s = Speaker(**d)
    s.save()
    return s

def add_all_speakers(restrict_to_dutch = True):
    header, data = open_file()
    speakers = []
    for line in data:
        d = make_speaker_dict(line)
        if restrict_to_dutch:
            if d['speaker_id'].startswith('V'): continue
        print(d['speaker_id'])
        speakers.append(add_speaker(line))
    return speakers



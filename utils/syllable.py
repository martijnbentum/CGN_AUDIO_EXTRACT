from utils import locations

cgn_lexicon_fn = locations.syllable_dir + 'cgnlex_2.0.txt'
frequency_list_fn = locations.syllable_dir + 'totrank.frq'


def load_cgn_lexicon():
    with open(cgn_lexicon_fn) as fin:
        t = fin.read()
    lexicon = [l.split('\\') for l in t.split('\n')]
    return lexicon

def load_word_frequency_dict():
    word_frequency_dict= {}
    with open(frequency_list_fn) as fin:
        t = fin.read()
    for l in t.split('\n')[2:]:
        line = [c for c in l.split(' ') if c]
        if len(line) != 3: continue
        frequency, word = line[-2:]
        frequency = int(frequency)
        word_frequency_dict[word] = frequency
    return word_frequency_dict

def line_to_ort_and_syl(line):
    if line[1] and line[10]:
        return line[1], line[10]
    return False, False

def make_ort_and_syl():
    lexicon = load_cgn_lexicon()
    ort_and_syl = []
    for line in lexicon:
        if not line or len(line) < 11: continue
        ort,syl = line_to_ort_and_syl(line)
        if ort and syl:
            syllables = syl.split('-')
            ort_and_syl.append([ort,syllables])
    return ort_and_syl

def syllable_frequency():
    '''creates a dictionary with syllables and corresponding frequencies.
    the orthographic word in cgnlex does not correspond with the word
    in the totrank, because they use different method to write diacritic
    characters. Therefore those words and there syllables are not taken
    into account.
    '''
    errors = []
    syllable_frequency_dict = {}
    ort_and_syl = make_ort_and_syl()
    word_frequency_dict = load_word_frequency_dict()
    for word, syllables in ort_and_syl:
        if word not in word_frequency_dict.keys():
            errors.append(word)    
            continue
        frequency = word_frequency_dict[word]
        for syllable in syllables:
            syllable = syllable.strip("'")
            if syllable not in syllable_frequency_dict.keys(): 
                syllable_frequency_dict[syllable] = frequency
            else: syllable_frequency_dict[syllable] += frequency
    return syllable_frequency_dict, errors
        




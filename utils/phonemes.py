import gruut_ipa

def make_dict(keys,values):
    d = {}
    for key, value in zip(keys,values):
        if key in d.keys(): d[key] += ', ' + value
        else: d[key] = value
    return d

def reverse_dict(input_d):
    d = {}
    for key, value in input_d.items():
        if ', ' in value:
            for v in value.split(', '):
                d[v]=key
        else: d[value]=key
    return d


class Sampa:
    plosives = 'p,t,b,d,k,g'.split(',')
    plosives_examples = '[p]ut,[b]ad,[t]ak,[d]ak,[k]at,[g]oal'.split(',')

    fricatives = 'f,v,s,z,S,Z,x,G,h'.split(',')
    fricatives_examples = '[f]iets,[v]at,[s]ap,[z]at,[sj]aal,rava[g]e,li[ch]t'
    fricatives_examples += ',re[g]en,ge[h]eel'
    fricatives_examples = fricatives_examples.split(',')

    sonorants = 'N,m,n,J,l,r,w,j'.split(',')
    sonorants_examples = 'la[ng],[m]at,[n]at,ora[nj]e,[l]at,[r]at'
    sonorants_examples += ',[w]at,[j]as'
    sonorants_examples = sonorants_examples.split(',')

    shortvowels = 'I,E,A,O,Y'.split(',')
    shortvowels_examples = 'l[i]p,l[e]g,l[a]t,b[o]m,p[u]t'.split(',')

    longvowels = 'i,y,e,2,a,o,u'.split(',')
    longvowels_examples = 'l[ie]p,b[uu]r,l[ee]g,d[eu]k,l[aa]t'
    longvowels_examples += ',b[oo]m,b[oe]k'
    longvowels_examples = longvowels_examples.split(',')

    sjwa = ['@']
    sjwa_examples = ['g[e]lijk']

    diphthongs = 'E+,Y+,A+'.split(',')
    diphthongs_examples = 'w[ij]s,h[ui]s,k[ou]d'.split(',')
    
    simple_diphthongs = '3,4,5'.split(',')

    loan_vowels = 'E:,Y:,O:'.split(',')
    loan_vowels_examples = 'sc[è]ne,fr[eu]le,z[o]ne'.split(',')

    simple_loan_vowels = 'E,Y,O'.split(',')

    nasal_vowels = 'E~,A~,O~,Y~'.split(',')
    nasal_vowels_examples = 'vacc[in],croiss[ant],c[on]gé,parf[um]'.split(',')

    simple_nasal_vowels = 'E,A,O,Y'.split(',')

    consonants = plosives + fricatives + sonorants
    consonants_examples = []
    consonants_examples.extend(plosives_examples)
    consonants_examples.extend(fricatives_examples)
    consonants_examples.extend(sonorants_examples)

    vowels = shortvowels+longvowels+sjwa+diphthongs+loan_vowels+nasal_vowels
    simple_vowels = shortvowels+longvowels+sjwa
    simple_vowels += simple_diphthongs+simple_loan_vowels+simple_nasal_vowels
    vowels_examples = []
    vowels_examples.extend(shortvowels_examples)
    vowels_examples.extend(longvowels_examples)
    vowels_examples.extend(sjwa_examples)
    vowels_examples.extend(diphthongs_examples)
    vowels_examples.extend(loan_vowels_examples)
    vowels_examples.extend(nasal_vowels_examples)

    symbols = consonants + vowels
    simple_symbols = consonants + simple_vowels
    examples = consonants_examples + vowels_examples

    consonants_to_examples = make_dict(consonants, consonants_examples)
    vowels_to_examples = make_dict(vowels, vowels_examples)
    symbols_to_examples = make_dict(symbols, examples)
    simple_symbols_to_examples = make_dict(simple_symbols, examples)
    
    @property
    def to_ipa_dict(self):
        if hasattr(self,'_to_ipa_dict'): return self._to_ipa_dict
        ipa = Ipa()
        examples_to_symbols = reverse_dict(self.symbols_to_examples)
        ipa_examples_to_symbols = reverse_dict(ipa.symbols_to_examples)
        d = {}
        for example, sampa_symbol in examples_to_symbols.items():
            ipa_symbol = ipa_examples_to_symbols[example]
            d[sampa_symbol] = ipa_symbol
        self._to_ipa_dict = d
        return self._to_ipa_dict

    @property
    def to_simple_ipa_dict(self):
        if hasattr(self,'_to_simple_ipa_dict'): return self._to_simple_ipa_dict
        ipa = Ipa()
        examples_to_symbols = reverse_dict(self.symbols_to_examples)
        ipa_examples_to_symbols = reverse_dict(ipa.simple_symbols_to_examples)
        d = {}
        for example, sampa_symbol in examples_to_symbols.items():
            ipa_symbol = ipa_examples_to_symbols[example]
            d[sampa_symbol] = ipa_symbol
        self._to_simple_ipa_dict = d
        return self._to_simple_ipa_dict

    @property
    def to_simple_sampa_dict(self):
        if hasattr(self,'_to_simple_sampa_dict'): 
            return self._to_simple_sampa_dict
        d = {}
        for symbol, simple_symbol in zip(self.symbols, self.simple_symbols):
            d[symbol] = simple_symbol
        self._to_simple_sampa_dict = d
        return self._to_simple_sampa_dict

    @property
    def to_cv_dict(self):
        if hasattr(self,'_to_cv_dict'): return self._to_cv_dict
        d = {}
        for symbol in self.symbols_to_examples.keys():
            if symbol in self.consonants: d[symbol] = 'C'
            elif symbol in self.vowels: d[symbol] = 'V'
            else: d[symbol] = ' '
        self._to_cv_dict = d
        return self._to_cv_dict
    
    @property
    def simple_sampa_to_simple_ipa(self):
        if hasattr(self,'_simple_sampa_to_simple_ipa_dict'): 
            return self._simple_sampa_to_simple_ipa_dict
        ipa = Ipa()
        examples_to_symbols = reverse_dict(self.simple_symbols_to_examples)
        ipa_examples_to_symbols = reverse_dict(ipa.simple_symbols_to_examples)
        d = {}
        for example, sampa_symbol in examples_to_symbols.items():
            ipa_symbol = ipa_examples_to_symbols[example]
            d[sampa_symbol] = ipa_symbol
        self._simple_sampa_to_simple_ipa_dict= d
        return self._simple_sampa_to_simple_ipa_dict

            
    @property
    def to_simple_sample_dict(self):
        pass
        

class Ipa:
    plosives = 'p,t,b,d,k,g'.split(',')
    plosives_examples = '[p]ut,[b]ad,[t]ak,[d]ak,[k]at,[g]oal'.split(',')

    fricatives = 'f,v,s,z,ʃ,ʒ,x,ɣ,h'.split(',')
    fricatives_examples = '[f]iets,[v]at,[s]ap,[z]at,[sj]aal,rava[g]e,li[ch]t'
    fricatives_examples += ',re[g]en,ge[h]eel'
    fricatives_examples = fricatives_examples.split(',')

    sonorants = 'ŋ,m,n,ɲ,l,r,w,j'.split(',')
    sonorants_examples = 'la[ng],[m]at,[n]at,ora[nj]e,[l]at,[r]at'
    sonorants_examples += ',[w]at,[j]as'
    sonorants_examples = sonorants_examples.split(',')

    shortvowels = 'ɪ,ɛ,ɑ,ɔ,ʏ'.split(',')
    shortvowels_examples = 'l[i]p,l[e]g,l[a]t,b[o]m,p[u]t'.split(',')

    longvowels = 'i,y,e,ø,a,o,u'.split(',')
    longvowels_examples = 'l[ie]p,b[uu]r,l[ee]g,d[eu]k,l[aa]t'
    longvowels_examples += ',b[oo]m,b[oe]k'
    longvowels_examples = longvowels_examples.split(',')

    sjwa = ['ə']
    sjwa_examples = ['g[e]lijk']

    diphthongs = 'ɛi,œy,ʌu'.split(',')
    diphthongs_examples = 'w[ij]s,h[ui]s,k[ou]d'.split(',')

    loan_vowels = 'ɛː,œː,ɔː'.split(',')
    loan_vowels_examples = 'sc[è]ne,fr[eu]le,z[o]ne'.split(',')

    simple_loan_vowels = 'ɛ,ʏ,ɔ'.split(',')

    nasal_vowels = 'ɛ̃ː,ɑ̃ː,ɔ̃ː,ʏ'.split(',')
    nasal_vowels_examples = 'vacc[in],croiss[ant],c[on]gé,parf[um]'.split(',')

    simple_nasal_vowels = 'ɛ,ɑ,ɔ,ʏ'.split(',')

    consonants = plosives + fricatives + sonorants
    consonants_examples = []
    consonants_examples.extend(plosives_examples)
    consonants_examples.extend(fricatives_examples)
    consonants_examples.extend(sonorants_examples)
    
    vowels = shortvowels+longvowels+sjwa+diphthongs+loan_vowels+nasal_vowels
    vowels_examples = []
    vowels_examples.extend(shortvowels_examples)
    vowels_examples.extend(longvowels_examples)
    vowels_examples.extend(sjwa_examples)
    vowels_examples.extend(diphthongs_examples)
    vowels_examples.extend(loan_vowels_examples)
    vowels_examples.extend(nasal_vowels_examples)

    simple_vowels = shortvowels+longvowels+sjwa+diphthongs
    simple_vowels += simple_loan_vowels+simple_nasal_vowels

    symbols = consonants + vowels
    examples = consonants_examples + vowels_examples

    simple_symbols = consonants + simple_vowels

    d = {c:ce for c, ce in zip(consonants,consonants_examples)}

    consonants_to_examples = make_dict(consonants,consonants_examples)
    vowels_to_examples = make_dict(vowels,vowels_examples)
    symbols_to_examples = make_dict(symbols,examples)

    simple_vowels_to_examples = make_dict(simple_vowels,vowels_examples)
    simple_symbols_to_examples = make_dict(simple_symbols, examples)

    @property
    def to_sampa_dict(self):
        if hasattr(self,'_to_sampa_dict'): return self._to_sampa_dict
        sampa = Sampa()
        self._to_sampa_dict = reverse_dict(sampa.to_ipa_dict)
        return self._to_sampa_dict

    @property
    def to_sampa_from_simple_ipa_dict(self):
        if hasattr(self,'_to_sampa_from_simple_ipa_dict'): 
            return self._to_sampa_from_simple_ipa_dict
        sampa = Sampa()
        d = reverse_dict(sampa.to_simple_ipa_dict)
        self._to_sampa_from_simple_ipa_dict= d
        return self._to_sampa_from_simple_ipa_dict

    @property
    def to_ipa_dict(self):
        if hasattr(self,'_to_ipa_dict'): return self._to_ipa_dict
        sampa = Sampa()
        self._to_ipa_dict = sampa.to_ipa_dict
        return self._to_ipa_dict

    @property
    def to_simple_ipa_dict(self):
        if hasattr(self,'_to_simple_ipa_dict'): return self._to_simple_ipa_dict
        sampa = Sampa()
        self._to_simple_ipa_dict= sampa.to_simple_ipa_dict
        return self._to_simple_ipa_dict

    @property
    def to_cv_dict(self):
        if hasattr(self,'_to_cv_dict'): return self._to_cv_dict
        d = {}
        for symbol in self.symbols_to_examples.keys():
            if symbol in self.consonants: d[symbol] = 'C'
            elif symbol in self.vowels: d[symbol] = 'V'
            else: d[symbol] = ' '
        self._to_cv_dict = d
        return self._to_cv_dict

    def to_ipa(self,sampa_symbol):
        return self.to_ipa_dict[sampa_symbol]

    def to_simple_ipa(self,sampa_symbol):
        if sampa_symbol in self.to_simple_ipa_dict.keys():
            return self.to_simple_ipa_dict[sampa_symbol]
    
    
    def word_to_simple_ipa(self,word):
        o = []
        for p in word.sampa_phonemes:
            ipa_phoneme = self.to_simple_ipa(p)
            if ipa_phoneme:
                o.append(ipa_phoneme)
        if not o: return
        word.word_ipa_phoneme = ''.join(o)
        word.ipa = ','.join(o)
        word.save()


def convert_simple_sampa_vocab_to_ipa_symbols(simple_sampa_vocab):
    sampa = Sampa()
    d = sampa.simple_sampa_to_simple_ipa
    nvocab = {}
    for key, value in simple_sampa_vocab.items():
        if key not in d.keys(): nvocab[key] = value
        else: nvocab[ d[key] ] = value
    return nvocab
        
    
def reorder_simple_ipa_vocab_to_phoneme_classes(simple_ipa_vocab):
    ipa = Ipa()
    out_vocab = {}
    new_indices = []
    symbols = []
    for s in ipa.simple_symbols + [' ']:
        if s not in symbols:symbols.append(s)
    for i,symbol in enumerate(symbols):
        for key, value in simple_ipa_vocab.items():
            if symbol == key:
                new_indices.append(value)
                out_vocab[key] = i
    return out_vocab, new_indices





    
    

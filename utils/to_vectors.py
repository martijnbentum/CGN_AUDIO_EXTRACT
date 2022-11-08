import os
import sys, os
import torch
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Model
from .wav2vec2_model import load_vocab
from . import audio
from . import locations

repo_dir = '/vol/tensusers/mbentum/CGN_AUDIO_EXTRACT/repo/'
default_checkpoint = repo_dir + 'o_first_test/checkpoint-6300'

pretrained_small="facebook/wav2vec2-xls-r-300m"
pretrained_big ="facebook/wav2vec2-xls-r-2b"

def checkpoint_to_processor_and_ctc_model(checkpoint):
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model =Wav2Vec2ForCTC.from_pretrained(checkpoint)
    return processor, model

def load_pretrained_processor_model(version = 'small'):
    '''
    # does not work
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-2b" ,
        cache_dir = locations.cache_dir)
    '''
    if version == 'small': name = pretrained_small
    else: name = pretrained_big
    # loading a non specific processor do not know whether this is ok
    processor = Wav2Vec2Processor.from_pretrained(default_checkpoint)
    model = Wav2Vec2Model.from_pretrained(name,
        cache_dir = locations.cache_dir)
    return processor, model

def audio_to_ctc_outputs(audio_filename, start=None, end=None, processor=None, 
    model=None):
    if not processor or not model: 
        print('loading default checkpoint trained cgn o:',default_checkpoint)
        processor, model = checkpoint_to_processor_and_model(default_checkpoint)
    array = audio.load_audio(audio_filename,start, end)
    inputs = processor(array, sampling_rate=16_000, return_tensors='pt',
        padding= True)
    with torch.no_grad():
        outputs = model(**inputs,output_hidden_states = True)
    return outputs

def audio_to_pretrained_outputs(audio_filename, start=None, end=None, 
    processor=None, model=None):
    if not processor or not model: 
        print('loading pretrained model: facebook/wav2vec2-xls-r-2b')
        processor, model = load_pretrained_processor_model()
    array = audio.load_audio(audio_filename,start, end)
    inputs = processor(array, sampling_rate=16_000, return_tensors='pt',
        padding= True)
    with torch.no_grad():
        outputs = model(**inputs,output_hidden_states = True)
    return outputs
    

def junk():
    directory = '/vol/bigdata2/corpora2/CGN2/data/audio/wav/comp-o/nl'

    MODEL_ID = "/vol/tensusers/lboves/huggingface_finetuned_with-mask/checkpoint-300"

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    tokenizer = Wav2Vec2CTCTokenizer("/vol/tensusers/lboves/HuggingFace/cgn_vocab.json", 
        unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    print('tokenizer created')

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)


    phone_list = processor.tokenizer.get_vocab()
    index2phone = {}
    for key in phone_list:
      index2phone[phone_list[key]] = key

    all_phones=[]
    for i in range(0, 38):
      all_phones.append(index2phone[i])


    maxnfiles = 500
    nfiles = 0
    nvectors = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            nfiles+=1
            print(nvectors)
        if (nfiles <= maxnfiles):
            print(f)
        speech_array, sampling_rate = librosa.load(f, sr=None, mono=False)
        speech_array16 = speech_array
        inputs = processor(speech_array16[0:16000*100], sampling_rate=16_000, 
        return_tensors="pt", padding=True)                                                                                                      #
        model_output = model(inputs.input_values, 
            attention_mask=inputs.attention_mask, output_hidden_states=True)                                                                                                    #
        logits = model_output.logits[-1,:,:].detach().numpy()
        #model_output.hidden_states
        #model_output.hidden_states[24].shape
        #
        for frameid in range(0, logits.shape[0]):
            #s = np.argsort(-1*logits[frameid,:].detach().numpy())
            s = np.argsort(-1*logits[frameid,:])
            if ((phone_list["a"] == s[0]) & (logits[frameid, s[0]] > 4)):
            #if (phone_list["a"] == s[0]):
                nvectors+=1
                M0+=1
                tmp=model_output.hidden_states[24][-1,frameid,:].detach().numpy()
                M1+=tmp
                M2+=np.square(tmp)


        np.save("Moments.npy", M0, M1, M2)

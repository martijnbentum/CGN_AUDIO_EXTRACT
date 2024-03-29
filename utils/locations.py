cgn_dir = '/vol/bigdata/corpora2/CGN2/'
cgn_audio_dir = cgn_dir + 'data/audio/wav/'
cgn_annot_dir = cgn_dir + 'data/annot/text/'
cgn_phoneme_dir = cgn_annot_dir + 'fon/'
cgn_ort_dir = cgn_annot_dir + 'ort/'
cgn_awd_dir = cgn_annot_dir + 'awd/'
cgn_fon_dir = cgn_annot_dir + 'fon/'
cgn_speaker_file = cgn_dir + 'data/meta/text/speakers.txt'
local_awd = '../awd/'
local_fon= '../fon/'
local_ort= '../ort/'

syllable_dir= '../SYLLABLE/'
timestamps_dir='../O_PHONEMES_TIMESTAMPS/'
ctc_hidden_states_dir='../O_CTC_HIDDEN_STATES/'
hidden_states_dir='../O_HIDDEN_STATES/'
new_ctc_hidden_states_dir='/vol/tensusers3/mbentum/O_CTC_HIDDEN_STATES/'
new_hidden_states_dir='/vol/tensusers3/mbentum/O_HIDDEN_STATES/'

kl_audio_dir = '/vol/tensusers3/mbentum/KL_AUDIOS/'

cache_dir = '../WAV2VEC_DATA/'
perceptron_dir = '../PERCEPTRONS/'

# ld
ld_base = '/vol/tensusers/mbentum/INDEEP/LD/'
mald_variable_stress = ld_base + 'mald_variable_stress_syllable/'
mald_variable_stress_wav = mald_variable_stress + 'wav_16khz/'
mald_variable_stress_occlusions_wav = mald_variable_stress 
mald_variable_stress_occlusions_wav += 'occlusions_wav_16khz/'
mald_variable_stress_pretrain_vectors = mald_variable_stress 
mald_variable_stress_pretrain_vectors += 'pretrain_wav2vec_vectors/'
mald_variable_stress_occlusions_pretrain_vectors = mald_variable_stress
mald_variable_stress_occlusions_pretrain_vectors += 'occlusions_pretrain'
mald_variable_stress_occlusions_pretrain_vectors += '_wav2vec_vectors/'

mald_word_recordings = ld_base + 'MALD/recordings/words_16khz/'
mald_pretrain_vectors = ld_base + 'MALD/pretrain_wav2vec_vectors/'
mald_code_vector_indices = ld_base + 'MALD/codevector_indices/'

mald_variable_stress_info = mald_variable_stress 
mald_variable_stress_info += 'mald_variable_stress_syllable.tsv'

mald_all_stress_info = ld_base + 'mald_all_words_stress_syllable.tsv'

all_words_stress_perceptron_dir = ld_base + 'ALL_WORDS_PERCEPTRONS/'
stress_perceptron_dir = ld_base + 'PERCEPTRONS/'
cnn_tf_comparison_dir = stress_perceptron_dir + 'CNN_TF_COMPARISON/'


codebook_indices_dir = '../CODEBOOK_INDICES/'
codebook_indices_phone_counts_dir = '../CODEBOOK_INDICES'
codebook_indices_phone_counts_dir += '_PHONE_COUNTS/'

leave_one_out_dir = ld_base +'LEAVE_ONE_OUT_PERCEPTRONS/'
leave_one_in_dir = ld_base +'LEAVE_ONE_IN_PERCEPTRONS/'

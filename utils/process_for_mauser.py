from pathlib import Path
import json

root_dir = Path('/vol/tensusers/mbentum/mauser/cgn/')

def make_transcription_files(component = 'k'):
    from text.models import Component
    k = Component.objects.get(name=component)
    component_dir = root_dir / f'comp-{component}'
    if not component_dir.exists(): component_dir.mkdir()
    transcriptions_dir = component_dir / 'transcriptions'
    if not transcriptions_dir.exists(): 
        transcriptions_dir.mkdir()
        textgrid_dir = component_dir / 'textgrids'
        if not textgrid_dir.exists(): textgrid_dir.mkdir()
    d = {}
    for audio in k.audio_set.all():
        print(audio.duration, audio.filename)
        d[ audio.cgn_id ] = audio.filename
        textgrid = audio.textgrid_set.all()[0]
        transcription = ' '.join([w.awd_word for w in textgrid.word_set.all()])
        filename = transcriptions_dir / '{}.txt'.format(textgrid.cgn_id)
        with open(filename, 'w') as f:
            f.write(transcription)
    file_map_filename = root_dir / f'file_map_{component}.json'
    with open(root_dir / file_map_filename, 'w') as f:
        json.dump(d, f)


from pathlib import Path
import os
from tqdm import tqdm
from phonemizer import phonemize
from joblib import Parallel, delayed
from functools import partial

langs = [
    ("English", 'en-us'),
]
root = '/home/henryhenrychen/DATA/corpus/LibriSpeech/LibriSpeech'
meta_dir = 'data'

def get_g2p_fn(code):
    return partial(phonemize.phonemize,
            backend='espeak',
            language=code,
            language_switch='remove-flags')


Path(meta_dir).mkdir(exist_ok=True)
for lang, code in langs:
    # Create metadata
    meta = []
    for sub in ['train-clean-100', 'dev-clean', 'test-clean']:
        for p1 in Path(root, sub).iterdir():
            if not p1.is_dir():
                continue
            for p2 in p1.iterdir():
                txt_name = '-'.join(str(p2).split('/')[-2:]) + '.trans.txt'
                with open(Path(p2, txt_name), 'r') as f:
                    data = {}
                    for line in f:
                        idx, txt = line.split(' ', 1)
                        data[idx] = txt.strip().lower()
                for wav_path in p2.rglob('*.wav'):
                    name = Path(wav_path).stem
                    txt = data.get(name, None)
                    if not txt: raise ValueError
                    idx = '/'.join(str(wav_path).split('/')[-4:]).strip('.wav')
                    meta.append([idx, sub.split('-')[0], txt])

    g2p_fn = get_g2p_fn(code)
    print(f'Start g2p {lang} ......')
    ipas = Parallel(n_jobs=4)(delayed(g2p_fn)(seg[-1]) for seg in tqdm(meta))
    meta = list(map(lambda x, y: '|'.join(x+[y]), meta, ipas))
    #meta = ['|'.join(x) for x in meta]
    out_name = Path(meta_dir, lang+'.txt')
    with open(out_name, 'w') as f:
        f.write("file|split|txt|ipa" + '\n')
        for line in meta:
            f.write(line+'\n')

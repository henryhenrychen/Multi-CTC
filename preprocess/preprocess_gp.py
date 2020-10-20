from pathlib import Path
import os
from tqdm import tqdm
from phonemizer import phonemize
from joblib import Parallel, delayed
from functools import partial

langs = [
    #("German", 'de'),
    ("French", 'fr-fr'),
    #("Czech", 'cs'),
    #("Spanish", 'es')
]
root = 'GlobalPhone'
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
    wav_dir = Path(root, lang, 'wav')
    for sub in ['train', 'dev', 'test']:
        with open(Path(root, lang, 'material', f"{sub}.word.text"), 'r') as f:
            data = f.read().splitlines()
        for line in data:
            line = line.split(" ", 1)
            if len(line) == 1 :
                continue
            idx, txt = line
            if not Path(wav_dir, idx + '.wav').is_file():
                continue
            #ipa = g2p(txt, code)
            #meta.append('|'.join([idx, sub, txt, ipa]))
            meta.append([idx, sub, txt])
    g2p_fn = get_g2p_fn(code)
    print(f'Start g2p {lang} ......')
    ipas = Parallel(n_jobs=4)(delayed(g2p_fn)(seg[-1]) for seg in tqdm(meta))
    meta = list(map(lambda x, y: '|'.join(x+[y]), meta, ipas))
    out_name = Path(meta_dir, lang+'.txt')
    with open(out_name, 'w') as f:
        f.write("file|split|txt|ipa" + '\n')
        for line in meta:
            f.write(line+'\n')

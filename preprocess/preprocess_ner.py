from pathlib import Path
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import partial
import random
import re

root = '/home/henryhenrychen/DATA/corpus/NER-Trs-Vol1'
meta_dir = 'data'

from opencc import OpenCC
from dragonmapper import hanzi
from czi import Converter_zh_ipa

UNWANTED_NAMES = set([
    "GJ_20160215_031",
    "CS_20160414_047",
    "CS_20160414_057",
    "CS_20160414_006",
    "CS_20160414_048",
    "CS_20160414_049",
    "CS_20160414_028",
    "CS_20160414_054",
    "CS_20160414_032",
    "CS_20160414_029",
    "CS_20160414_025",
    "CS_20160414_043",
    "CS_20160414_052",
    "CS_20160414_022",
    "CS_20160414_023",
    "CS_20160414_068",
    "CS_20160414_073",
    "CS_20160407_029",
    "CS_20160407_028",
    "CS_20160407_021",
    "CS_20160407_037",
    "CS_20160519_132",
    "CS_20160519_137",
    "CS_20160519_047",
    "CS_20160519_114",
    "CS_20160519_115",
    "CS_20160519_126",
    "CS_20160602_029",
    "CS_20160526_165",
    "CS_20160526_044",
    "CS_20160505_025",
    "CS_20160505_139",
    "CS_20160414_013",
    "CS_20160505_024",
    "CS_20160225_012",
    "CS_20160616_059",
    "CS_20160421_155"

    ])

def word_preprocess(text):
    text = text.replace('s ', '')
    return text

def zhuyin_preprocess(text):
    text = text.replace('d', 'ㄉㄧ')
    text = text.replace('i', 'ㄛ')
    text = text.replace('ń', 'ㄣ˙')
    return text

class TW_to_IPA_Converter:
    def __init__(self):
        self.cc = OpenCC('tw2sp')
        self.zhuyin_ipa_converter = Converter_zh_ipa("pinyinbiao_ntu.txt")

    def go(self, text):
        sp_txt = self.cc.convert(text)
        sp_txt = word_preprocess(sp_txt)

        zhuyin = hanzi.to_zhuyin(sp_txt)
        zhuyin = zhuyin_preprocess(zhuyin)

        ipa = self.zhuyin_ipa_converter.zh2ipa(zhuyin, taiwan=False)
        # strip tone
        ipa = re.sub(r'\d+', '', ipa)

        return ipa

g2p_fn = TW_to_IPA_Converter().go

Path(meta_dir).mkdir(exist_ok=True)
lang = 'Chinese'
meta = []
for line, wav_path in enumerate(Path(root, 'Train/Clean').rglob('*.wav')):
    txt_path = str(wav_path).replace('Wav', 'Text').replace('.wav', '.txt')
    if not Path(txt_path).is_file():
        print(f'{txt_path} cannot found')
        continue
    if wav_path.stem in UNWANTED_NAMES:
        #print(f"{wav_path} filtered by defined unwanted names")
        continue
    txt = Path(txt_path).read_text().strip()#.replace(' ', '')
    idx = '/'.join(str(wav_path).split('/')[-6:]).strip('.wav')
    meta.append([idx, txt])
    print(line, idx, txt)
    g2p_fn(txt)

print(f'Start g2p {lang} ......')
#ipas = Parallel(n_jobs=4)(delayed(g2p_fn)(seg[-1]) for seg in tqdm(meta))
ipas = [g2p_fn(seg[-1]) for seg in tqdm(meta)]

# Sample to train, dev, test
train_num = int(len(meta) * 0.9)
dev_num = (len(meta) - train_num) // 2
splits = ['train'] * train_num + ['dev'] * dev_num + ['test'] * dev_num
random.shuffle(splits)


# Concat all
meta = list(map(lambda x, y, z: '|'.join([x[0]] + [z] + [x[1]]+[y]), meta, ipas, splits))

#meta = ['|'.join(x) for x in meta]
out_name = Path(meta_dir, lang+'.txt')
with open(out_name, 'w') as f:
    f.write("file|split|txt|ipa" + '\n')
    for line in meta:
        f.write(line+'\n')

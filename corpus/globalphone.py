from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import pdb
import math


code2weight = {
    'FR': 25,
    'GE': 15,
    'SP': 17.5,
    'CZ': 27
        }
code2path = {
    'FR': 'French',
    'GE': 'German',
    'SP': 'Spanish',
    'CZ': 'Czech'
}
# TODO
LIBRI_ROOT = '/home/henryhenrychen/DATA/corpus/LibriSpeech/LibriSpeech'
def name2path(root, name):
    if name[:2] in code2path:
        path = Path(root, code2path[name[:2]], 'wav', name + '.wav')
    else:
        path = Path(LIBRI_ROOT, name + '.wav')
    return path


class GPDataset(Dataset):
    def __init__(self, tokenizer, root, meta, target, split, bucket, split_frac=1):
        # Setup
        self.tokenizer = tokenizer
        self.bucket_size = bucket
        self.test = split in ['test', 'dev']
        if split in ['dev', 'test']:
            assert split_frac == 1, "Should not sample from dev or test data"
        if len(meta) != 1 and split_frac != 1:
            raise ValueError('only support sample training set for transfer')

        data = []
        for m in meta:
            if '|' in m:
                o = pd.read_csv(m.split('|')[0], sep='|')
                o = o[o[target].notnull()]
                o = o[o['split'] == split]
                d = o.sample(frac=float(m.split('|')[1]))
                print(f"Data {m} split {split} from {len(o)} to {len(d)}")
            else:
                d = pd.read_csv(m, sep='|')
                d = d[d[target].notnull()]
                d = d[d['split'] == split]
            data.append(d)
        data = pd.concat(data, ignore_index=True)
        #data = pd.concat([pd.read_csv(m, sep='|') for m in meta], ignore_index=True)
        meta = data[data[target].notnull()]
        meta = meta[meta['split'] == split]

        meta = meta.sample(frac=split_frac)

        file_list = [name2path(root, x) for x in list(meta['file'])]
        #file_list = list(meta['file'])
        assert target in ['txt', 'ipa']
        text = [tokenizer.encode(x) for x in list(meta[target])]



        # Sort by text length
        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text),
                                              reverse=True,
                                              key=lambda x:len(x[1]))])
        if self.test:
            self.buckets = []
            # Do bucketing first and traverse all data once
            Length = len(self.file_list)
            for b in range(math.ceil(Length / self.bucket_size)):
                i = b*self.bucket_size ; j = min(i + self.bucket_size, Length)
                self.buckets.append(list(zip(self.file_list[i:j], self.text[i:j])))


        #self.text = list(meta[target])

    def __getitem__(self, index):
        if self.test:
            return self.buckets[index]
        else:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]

    def __len__(self):
        if self.test:
            return len(self.buckets)
        else:
            return len(self.file_list)



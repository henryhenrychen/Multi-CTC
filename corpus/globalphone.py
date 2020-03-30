from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import pdb

code2path = {
    'FR': 'French',
    'GE': 'German',
    'SP': 'Spanish',
    'CZ': 'Czech'
}
def name2path(root, name):
    assert name[:2] in code2path
    return Path(root, code2path[name[:2]], 'wav', name + '.wav')


class GPDataset(Dataset):
    def __init__(self, tokenizer, root, meta, target, split, bucket, split_frac=1):
        # Setup
        self.tokenizer = tokenizer
        self.bucket_size = bucket

        data = pd.concat([pd.read_csv(m, sep='|') for m in meta], ignore_index=True)
        meta = data[data[target].notnull()]
        meta = meta[meta['split'] == split]

        if split in ['dev', 'test']:
            assert split_frac == 1, "Should not sample from dev or test data"

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


        #self.text = list(meta[target])

    def __getitem__(self, index):
        # Return a bucket
        index = min(len(self.file_list)-self.bucket_size, index)
        return [(f_path, txt) for f_path, txt in
                zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]

    def __len__(self):
        return len(self.file_list)



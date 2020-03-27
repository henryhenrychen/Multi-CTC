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
    def __init__(self, tokenizer, root, meta, target, split):
        # Setup
        self.tokenizer = tokenizer
        self.root = root
        data = pd.concat([pd.read_csv(m, sep='|') for m in meta], ignore_index=True)
        meta = data[data[target].notnull()]

        meta = meta[meta['split'] == split]
        self.file_list = list(meta['file'])
        assert target in ['txt', 'ipa']
        self.text = list(meta[target])

    def __getitem__(self, index):
        audio_path = name2path(self.root, self.file_list[index])
        text_seq = self.tokenizer.encode(self.text[index])
        return audio_path, text_seq

    def __len__(self):
        return len(self.file_list)



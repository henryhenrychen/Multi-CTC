import pandas as pd
from pathlib import Path

metas = [
    ('meta/French.txt', 'fr'),
    ('meta/German.txt', 'de'),
    ('meta/Czech.txt', 'cs'),
    ('meta/Spanish.txt', 'sp')
]

def generate_vocab(metas, target, output_dir):
    name = '_'.join(map(lambda x: x[1], metas)) + f'.{target}.txt'
    data = pd.concat([pd.read_csv(meta[0], sep='|') for meta in metas], ignore_index=True)
    data = data[data.ipa.notnull()]
    tar_list = data[data.split == 'train'][target]
    tar_set = sorted(set().union(*map(lambda x:set(list(x)), tar_list)))
    with open(Path(output_dir, name), 'w') as f:
        for line in tar_set:
            f.write(line + '\n')
    print(f'Saving {name} ...')
    return

if __name__ == '__main__' :
    output_dir = 'corpus'
    for target in ['ipa', 'txt']:
        for meta_list in [metas[:-1], metas[:], metas[-1:]]:
            generate_vocab(meta_list, target, output_dir)

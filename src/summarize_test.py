import yaml
import os
import argparse
from pathlib import Path
import pandas as pd
import torch
import math
from pathlib import Path
from itertools import groupby
import torch




# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--result_root', type=str)
parser.add_argument('--test_config', type=str)
paras = parser.parse_args()

def get_score(path):
    log = list(Path(path).glob('*.csv'))
    if not len(log):
        return 100
    with open(log[0], 'r') as f:
        lines = f.read().splitlines()
        score = float(lines[-1].strip().split('|')[-1])
    return score

def exec(args):
    method = args.test_config
    dirs = [Path(p, method) for p in Path(args.result_root).iterdir()]

    print("="*80)
    print(f"Root = {args.result_root}")
    print(f"Testing config = {method}")

    for frac, group in groupby(sorted(dirs, key=lambda x:x.parent.name.split('-')[1]),
            key=lambda x:x.parent.name.split('-')[1]):
        group_score = sorted([(path, get_score(path)) for path in group], key=lambda x:x[1])
        best, best_score = group_score[0]
        print('{:<30} | {:2.6f}'.format(str(best.parent.name), best_score))
    print("="*80)



if __name__ == '__main__':
    exec(paras)

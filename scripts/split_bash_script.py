import yaml
import os
import argparse
from pathlib import Path
import pandas as pd
import torch
import math



# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--script', type=str)
parser.add_argument('--device_num', type=int)
paras = parser.parse_args()


def exec(args):
    assert Path(args.script).is_file()
    with open(args.script, 'r') as f:
        data = f.read().splitlines()
    tmp_dir = Path('tmp')
    tmp_dir.mkdir(exist_ok=True)

    num = args.device_num
    seg = len(data) // num
    for i in range(num):
        with open(Path(tmp_dir, Path(args.script).name + f"{i}.txt"), 'w') as f:
            start = i * seg
            end = start + seg if i != num - 1 else len(data)
            for line in data[start:end]:
                f.write(line + '\n')






if __name__ == '__main__':
    exec(paras)

import yaml
import os
import argparse
from pathlib import Path
import pandas as pd
import torch
import math



# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--model_path', type=str)
parser.add_argument('--output_path', type=str, default='result/')
parser.add_argument('--test_config', type=str)
parser.add_argument('--cuda_device', type=int)
paras = parser.parse_args()



def run(output_dir, base_config, src_ckpt_path, cuda_device=None):
    src_ckpt_path = Path(src_ckpt_path)
    src_config_path = Path(src_ckpt_path.parent, 'config.yaml')
    if not src_config_path.is_file():
        # TODO for dealing with pretrain_mono
        src_config_path = Path(src_ckpt_path.parents[1], '_'.join(src_ckpt_path.parent.name.split('_')[:-1]) + '.yaml')
        #return

    config = yaml.load(open(base_config, 'r'), Loader=yaml.FullLoader)
    config['src']['ckpt'] = str(src_ckpt_path)
    config['src']['config'] = str(src_config_path)

    #cur_output_dir = Path(output_dir, src_ckpt_path.parents[0].name + f"_{Path(base_config).stem}")
    output_dir = Path(output_dir, src_ckpt_path.parents[0].name)
    output_dir.mkdir(exist_ok=True)
    cur_output_dir = Path(output_dir, Path(base_config).stem)
    cur_output_dir.mkdir(exist_ok=True)
    cur_config = Path(cur_output_dir, 'config.yaml')
    with open(cur_config, 'w') as f:
        yaml.dump(config, f)

    cmd = f"python main.py --test --config {cur_config} --name {Path(base_config).stem} --outdir {output_dir} --logdir {output_dir}"

    if cuda_device is not None:
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} " + cmd
    print(cmd)

def exec(args):
    model_path_name = Path(args.model_path).name
    cur_out_dir = Path(args.output_path, model_path_name)
    cur_out_dir.mkdir(exist_ok=True)

    for path in Path(args.model_path).rglob('best*'):
        run(cur_out_dir, args.test_config, path, args.cuda_device)




if __name__ == '__main__':
    exec(paras)

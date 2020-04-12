import yaml
import os
import argparse
from pathlib import Path
import pandas as pd
import torch
import math


VALID_EVERY_EPOCH = 5
TOTAL_EPOCH = 400

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--pretrain_path', type=str)
parser.add_argument('--output_path', type=str, default='ckpt/')
parser.add_argument('--logdir', type=str, default='log')
parser.add_argument('--pretrain_config', type=str)
parser.add_argument('--adapt_config', type=str)
parser.add_argument('--adapt_every_step', type=int)
parser.add_argument('--top_adapt_num', type=int, default=20)
parser.add_argument('--cuda_device', type=int)
paras = parser.parse_args()

def get_training_data_size(meta_file, target):
    meta = pd.read_csv(meta_file, sep='|')
    meta = meta[meta[target].notnull()]
    meta = meta[meta['split'] == 'train']
    return len(meta)

def get_cur_base_config(pretrain_config, adapt_config):
    pretrain_config = yaml.load(open(pretrain_config, 'r'), Loader=yaml.FullLoader)
    base_config = yaml.load(open(adapt_config, 'r'), Loader=yaml.FullLoader)
    base_config['data']['audio'] = pretrain_config['data']['audio']
    for k, v in pretrain_config['data']['text'].items():
        base_config['data']['transfer'][k] = v
    base_config['model'] = pretrain_config['model']
    return base_config


def run(pretrain_path, output_dir, log_dir, config, cuda_device=None):
    #config = yaml.load(open(base_config, 'r'), Loader=yaml.FullLoader)

    meta_file = config['data']['corpus']['metas']
    assert len(meta_file) == 1, "Should adapting to only 1 meta file"
    train_full_size = get_training_data_size(meta_file[0], config['data']['corpus']['target'])
    bs = config['data']['corpus'] ['batch_size']

    # Adjust pretrain model
    config['data']['transfer']['src_ckpt'] = str(pretrain_path)

    # Adjust target language size
    for frac in [0.015, 0.03, 0.06, 0.12, 0.5, 1]:
        # Adjust learning rate
        for lr in [0.1, 0.5, 1]:
            config['data']['corpus']['train_split'] = frac
            config['hparas']['lr'] = lr
            valid_step = train_full_size * frac / bs * VALID_EVERY_EPOCH
            valid_step = math.ceil(valid_step / 100) * 100 # round to 100
            max_step = train_full_size * frac / bs * TOTAL_EPOCH
            max_step = math.ceil(max_step / 100) *100
            config['hparas']['max_step'] = max_step
            config['hparas']['valid_step'] = valid_step

            step = int(str(pretrain_path.stem))
            cur_name = f"step{step}-frac{frac}-lr{lr}"
            cur_output_path = Path(output_dir, cur_name)
            cur_output_path.mkdir(exist_ok=True)
            cur_config = Path(cur_output_path, f"config.yaml")
            with open(cur_config, 'w') as f:
                yaml.dump(config, f)
            cmd = f"python main.py --transfer --config {cur_config} --name {cur_name} --ckpdir {output_dir} --logdir {log_dir}"

            if cuda_device is not None:
                cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} " + cmd
            print(cmd)

def exec(args):
    cur_name = f"{Path(args.pretrain_path).stem}_{Path(args.adapt_config).stem}"
    cur_out_dir = Path(args.output_path, cur_name)
    cur_out_dir.mkdir(exist_ok=True)
    cur_log_dir = Path(args.logdir, cur_name)
    cur_log_dir.mkdir(exist_ok=True)

    base_config = get_cur_base_config(args.pretrain_config, args.adapt_config)

    valids = []
    for pretrain_path in Path(args.pretrain_path).rglob('*0.path'):
        step = int(str(pretrain_path.stem))
        if args.adapt_every_step is None or step % args.adapt_every_step == 0:
            score = torch.load(pretrain_path)['wer']
            valids.append((pretrain_path, score))
    valids = sorted(valids, key=lambda x: x[1])[:args.top_adapt_num]
    for pretrain_path, _ in valids:
        #step = int(str(pretrain_path.stem))
        #if (args.adapt_every_step is None or step % args.adapt_every_step == 0)\
        #        and args.min_step <= step <= args.max_step :
        run(pretrain_path, cur_out_dir, cur_log_dir, base_config, args.cuda_device)




if __name__ == '__main__':
    exec(paras)

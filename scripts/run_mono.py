import yaml
import os
import argparse
from pathlib import Path
import math

MAX_EPOCH =
# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--base_config', type=str)
parser.add_argument('--output_path', type=str, default='ckpt/')
paras = parser.parse_args()



def run(output_path, frac, base_config):
    config = yaml.load(open(base_config, 'r'), Loader=yaml.FullLoader)
    config['data']['corpus']['train_split'] = frac
    valid_step = 6000*4/16*frac
    valid_step = math.ceil(valid_step / 100) * 100
    max_step = math.ceil(6000/16*frac) * 200 # 200 epoch
    config['hparas']['max_step'] = max_step
    config['hparas']['valid_step'] = valid_step

    cur_config = Path(output_path, f'frac{frac}_config.yaml')
    with open(cur_config, 'w') as f:
        yaml.dump(config, f)
    cmd = f"python main.py --config {cur_config}"
    print(cmd)
    #os.system(cmd)

def exec(args):
    for frac in [0.015, 0.03, 0.06, 0.12, 0.5, 1]:
        run(args.output_path, frac, args.base_config)




if __name__ == '__main__':
    exec(paras)

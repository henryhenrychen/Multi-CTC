import yaml
import os
import argparse
from pathlib import Path

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--pretrain_path', type=str)
parser.add_argument('--output_path', type=str, default='ckpt/')
parser.add_argument('--adaption_config', type=str)
parser.add_argument('--adapt_every', type=int, default=10000)
parser.add_argument('--cuda_device', type=int)
paras = parser.parse_args()



def run(pretrain_path, output_path, base_config, cuda_device=None):
    config = yaml.load(open(base_config, 'r'), Loader=yaml.FullLoader)
    config['data']['transfer']['src_ckpt'] = str(pretrain_path)
    step = int(str(pretrain_path.stem))
    name = f"{str(pretrain_path).split('/')[-2]}_{Path(base_config).stem}_step{step}"
    output_path = Path(output_path, name)
    output_path.mkdir(exist_ok=True)
    cur_config = Path(output_path, f"config_step{step}.yaml")
    with open(cur_config, 'w') as f:
        yaml.dump(config, f)
    cmd = f"python main.py --transfer --config {cur_config} --outdir {str(output_path)} --name {name}"
    if cuda_device is not None:
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} " + cmd
    print(cmd)
    os.system(cmd)

def exec(args):
    output_path = Path(args.output_path, Path(args.adaption_config).stem)
    output_path.mkdir(exist_ok=True)
    for pretrain_path in sorted(Path(args.pretrain_path).rglob('*0.path'), key=lambda p: int(str(p.stem))):
        step = int(str(pretrain_path.stem))
        if step % args.adapt_every == 0:
            run(pretrain_path, output_path, args.adaption_config, args.cuda_device)




if __name__ == '__main__':
    exec(paras)

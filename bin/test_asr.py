import copy
import torch
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed

from src.solver import BaseSolver
from src.asr import ASR
from src.data import load_dataset
from src.util import cal_er
import editdistance as ed

from tqdm import tqdm
from pathlib import Path

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)

        self.config['data'] = self.src_config['data']
        if 'batch_size' in self.config:
            self.config['data']['corpus']['batch_size'] = self.config['batch_size']
        self.config['data']['corpus']['mode'] = 'test'
        self.config['data'].pop('transfer')
        self.config['model'] = self.src_config['model']

        # Override batch size for beam decoding
        self.greedy = self.config['decode']['beam_size'] == 1
        if not self.greedy:
            raise NotImplementedError

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.dv_set, self.tt_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
            load_dataset(self.paras.njobs, self.paras.gpu,
                         self.paras.pin_memory, False, **self.config['data'])
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model '''
        # Model
        init_adadelta = self.src_config['hparas']['optimizer'] == 'Adadelta'
        self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **
                         self.config['model']).to(self.device)

        # Load target model in eval mode
        self.load_ckpt()


    def exec(self):
        ''' Testing End-to-end ASR system '''
        # Setup output
        method = 'beam' if not self.greedy else 'greedy'
        self.cur_output_path = Path(self.ckpdir, '{}_{}_output.csv'.format(method, self.config['data']['corpus']['target']))
        with open(self.cur_output_path, 'w') as f:
            f.write("hyp|truth\n")
        tt_output, tt_txt = [], []
        for data in tqdm(self.tt_set):
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                ctc_output, encode_len = self.model(feat, feat_len)

            ctc_output = [x[:length] for x, length in zip(ctc_output, encode_len)]
            if self.greedy:
                ctc_output = [x.argmax(dim=-1) for x in ctc_output]
            else:
                pass

            tt_output.extend(ctc_output)
            tt_txt.extend(txt)
        f = open(self.cur_output_path, 'a')
        er = []
        for hyp, truth in zip(tt_output, tt_txt):
            p = self.tokenizer.decode(hyp.tolist(), ignore_repeat=True)
            t = self.tokenizer.decode(truth.tolist())
            f.write(f"{p}|{t}\n")
            er.append(float(ed.eval(p, t)) / len(t))
        tt_error = sum(er) / len(er)
        f.write(f"ERROR|{float(tt_error)}\n")
        f.close()



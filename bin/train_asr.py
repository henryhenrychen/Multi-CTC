import torch
from src.solver import BaseSolver

from src.asr import ASR
from src.optim import Optimizer
from src.data import load_dataset
from src.util import human_format, cal_er, feat_to_fig
import pdb
import math

SAVE_EVERY = True

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        # Logger settings
        self.best_wer = {'ctc': 3.0}
        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        _, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt != 0, dim=-1)

        return feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.feat_dim, self.vocab_size, self.tokenizer, msg= \
            load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                         self.curriculum > 0, **self.config['data'])
        self.verbose(msg)

    def transfer_weight(self):
        # Load weights
        #self.model.init_ctclayer(new_vocab_size, self.device)
        msg = self.model.transfer_with_mapping(self.vocab_size,
                                self.config['data']['transfer'],
                                self.tokenizer)
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model and optimizer '''
        # Model
        init_adadelta = self.config['hparas']['optimizer'] == 'Adadelta'
        self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **
                         self.config['model']).to(self.device)
        self.verbose(self.model.create_msg())
        model_paras = [{'params': self.model.parameters()}]

        # Losses
        # Note: zero_infinity=False is unstable?
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)


        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.verbose(self.optimizer.create_msg())

        # Enable AMP if needed
        self.enable_apex()

        # Automatically load pre-trained model if self.paras.load is given
        if self.paras.load:
            self.load_ckpt()

        if self.paras.transfer:
            self.transfer_weight()

        # ToDo: other training methods

    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(
            human_format(self.max_step)))
        ctc_loss =  None
        n_epochs = 0
        self.timer.set()

        while self.step < self.max_step:
            # Renew dataloader to enable random sampling
            if self.curriculum > 0 and n_epochs == self.curriculum:
                self.verbose(
                    'Curriculum learning ends after {} epochs, starting random sampling.'.format(n_epochs))
                self.tr_set, _, _, _, _, _ = \
                    load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                                 False, **self.config['data'])
            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                tf_rate = self.optimizer.pre_step(self.step)
                total_loss = 0

                # Fetch data
                feat, feat_len, txt, txt_len = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward model
                # Note: txt should NOT start w/ <sos>
                ctc_output, encode_len = self.model(feat, feat_len)


                # Compute all objectives
                if self.paras.cudnn_ctc:
                    ctc_loss = self.ctc_loss(ctc_output.transpose(0, 1),
                                             txt.to_sparse().values().to(device='cpu', dtype=torch.int32),
                                             [ctc_output.shape[1]] *
                                             len(ctc_output),
                                             txt_len.cpu().tolist())
                else:
                    ctc_loss = self.ctc_loss(ctc_output.transpose(
                        0, 1), txt, encode_len, txt_len)


                total_loss = ctc_loss

                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(total_loss)

                self.step += 1

                # Logger
                if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
                                  .format(total_loss.cpu().item(), grad_norm, self.timer.show()))
                    #self.write_log('wer', {'tr_ctc': cal_er(self.tokenizer, ctc_output, txt, ctc=True)})
                    self.write_log('per', {'tr_ctc': cal_er(self.tokenizer, ctc_output, txt, mode='per', ctc=True)})
                    self.write_log(
                        'loss', {'tr_ctc': ctc_loss})

                # Validation
                if (self.step == 1) or (self.step % self.valid_step == 0):
                    self.validate()

                # End of step
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()
                self.timer.set()
                if self.step > self.max_step:
                    break
            n_epochs += 1
        self.log.close()

    def validate(self):
        # Eval mode
        self.model.eval()
        dev_wer = {'ctc': []}

        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                ctc_output, encode_len = self.model(feat, feat_len)

            dev_wer['ctc'].append(cal_er(self.tokenizer, ctc_output, txt, mode='cer', ctc=True))

            # Show some example on tensorboard
            if i == len(self.dv_set)//2:
                for i in range(min(len(txt), self.DEV_N_EXAMPLE)):
                    if self.step == 1:
                        self.write_log('true_text{}'.format(
                            i), self.tokenizer.decode(txt[i].tolist()))
                    self.write_log('ctc_text{}'.format(i), self.tokenizer.decode(ctc_output[i].argmax(dim=-1).tolist(),
                                                                                     ignore_repeat=True))

        # Ckpt if performance improves
        for task in ['ctc']:
            dev_wer[task] = sum(dev_wer[task])/len(dev_wer[task])
            if dev_wer[task] < self.best_wer[task]:
                self.best_wer[task] = dev_wer[task]
                self.save_checkpoint('best_{}.pth'.format(task), 'wer', dev_wer[task])
            self.write_log('per', {'dv_'+task: dev_wer[task]})
        self.save_checkpoint('latest.pth', 'wer', dev_wer['ctc'], show_msg=False)
        if self.paras.save_every:
            self.save_checkpoint(f'{self.step}.path', 'wer', dev_wer['ctc'], show_msg=False)

        # Resume training
        self.model.train()

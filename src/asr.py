import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.util import init_weights, init_gate
from src.module import VGGExtractor, CNNExtractor, RNNLayer


class ASR(nn.Module):
    ''' ASR CTC model '''

    def __init__(self, input_size, vocab_size, init_adadelta, encoder):
        super(ASR, self).__init__()

        # Setup
        self.vocab_size = vocab_size

        # Modules
        self.encoder = Encoder(input_size, **encoder)
        self.ctc_layer = nn.Linear(self.encoder.out_dim, vocab_size)

        # Init
        if init_adadelta:
            self.apply(init_weights)


    def create_msg(self):
        # Messages for user
        msg = []
        msg.append('Model spec.| Encoder\'s downsampling rate of time axis is {}.'.format(
            self.encoder.sample_rate))
        if self.encoder.vgg:
            msg.append(
                '           | VGG Extractor w/ time downsampling rate = 4 in encoder enabled.')
        if self.encoder.cnn:
            msg.append(
                '           | CNN Extractor w/ time downsampling rate = 4 in encoder enabled.')
        #msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(
        #    self.ctc_weight))
        return msg

    def forward(self, audio_feature, feature_len):
        '''
        Arguments
            audio_feature - [BxTxD] Acoustic feature with shape
            feature_len   - [B]     Length of each sample in a batch
        '''
        # Init
        bs = audio_feature.shape[0]
        ctc_output = None

        # Encode
        encode_feature, encode_len = self.encoder(audio_feature, feature_len)

        ctc_output = F.log_softmax(self.ctc_layer(encode_feature), dim=-1)

        return ctc_output, encode_len


class Encoder(nn.Module):
    ''' Encodes acoustic feature to latent representation, see config file for more details.'''

    def __init__(self, input_size, prenet, module, bidirection, dim, dropout, layer_norm, proj, sample_rate, sample_style):
        super(Encoder, self).__init__()

        # Hyper-parameters checking
        self.vgg = prenet == 'vgg'
        self.cnn = prenet == 'cnn'
        self.sample_rate = 1
        assert len(sample_rate) == len(dropout), 'Number of layer mismatch'
        assert len(dropout) == len(dim), 'Number of layer mismatch'
        num_layers = len(dim)
        assert num_layers >= 1, 'Encoder should have at least 1 layer'

        # Construct model
        module_list = []
        input_dim = input_size

        # Prenet on audio feature
        if self.vgg:
            vgg_extractor = VGGExtractor(input_size)
            module_list.append(vgg_extractor)
            input_dim = vgg_extractor.out_dim
            self.sample_rate = self.sample_rate*4
        if self.cnn:
            cnn_extractor = CNNExtractor(input_size, out_dim=dim[0])
            module_list.append(cnn_extractor)
            input_dim = cnn_extractor.out_dim
            self.sample_rate = self.sample_rate*4

        # Recurrent encoder
        if module in ['LSTM', 'GRU']:
            for l in range(num_layers):
                module_list.append(RNNLayer(input_dim, module, dim[l], bidirection, dropout[l], layer_norm[l],
                                            sample_rate[l], sample_style, proj[l]))
                input_dim = module_list[-1].out_dim
                self.sample_rate = self.sample_rate*sample_rate[l]
        else:
            raise NotImplementedError

        # Build model
        self.in_dim = input_size
        self.out_dim = input_dim
        self.layers = nn.ModuleList(module_list)

    def forward(self, input_x, enc_len):
        for _, layer in enumerate(self.layers):
            input_x, enc_len = layer(input_x, enc_len)
        return input_x, enc_len

from distutils.version import LooseVersion
import logging
import math
import random
import six

import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from espnet.nets.e2e_asr_common import end_detect

from espnet.nets.pytorch_backend.rnn.attentions import att_to_numpy

from espnet.nets.pytorch_backend.nets_utils import mask_by_length
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.rnn.beamsearch import BeamableModel
from espnet.nets.pytorch_backend.rnn.beamsearch import BeamSearch
from espnet.nets.scorer_interface import ScorerInterface

MAX_DECODER_OUTPUT = 5
CTC_SCORING_RATIO = 1.5


class Decoder(torch.nn.Module, ScorerInterface, BeamableModel):
    """Decoder module

    :param int eprojs: encoder projection units
    :param int odim: dimension of outputs
    :param str dtype: gru or lstm
    :param int dlayers: decoder layers
    :param int dunits: decoder units
    :param int sos: start of sequence symbol id
    :param int eos: end of sequence symbol id
    :param torch.nn.Module att: attention module
    :param int verbose: verbose level
    :param list char_list: list of character strings
    :param ndarray labeldist: distribution of label smoothing
    :param float lsm_weight: label smoothing weight
    :param float sampling_probability: scheduled sampling probability
    :param float dropout: dropout rate
    :param float context_residual: if True, use context vector for token generation
    :param bool replace_sos: use for multilingual (speech/text) translation
    """

    def __init__(self, eprojs, odim, dtype, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0,
                 dropout=0.0, context_residual=False, replace_sos=False, num_encs=1):

        torch.nn.Module.__init__(self)
        self.dtype = dtype
        self.dunits = dunits
        self.dlayers = dlayers
        self.context_residual = context_residual
        self.embed = torch.nn.Embedding(odim, dunits)
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            torch.nn.LSTMCell(dunits + eprojs, dunits) if self.dtype == "lstm" else torch.nn.GRUCell(dunits + eprojs,
                                                                                                     dunits)]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in six.moves.range(1, self.dlayers):
            self.decoder += [
                torch.nn.LSTMCell(dunits, dunits) if self.dtype == "lstm" else torch.nn.GRUCell(dunits, dunits)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]
            # NOTE: dropout is applied only for the vertical connections
            # see https://arxiv.org/pdf/1409.2329.pdf
        self.ignore_id = -1

        if context_residual:
            self.output = torch.nn.Linear(dunits + eprojs, odim)
        else:
            self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.odim = odim
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        self.num_encs = num_encs

        # for multilingual E2E-ST
        self.replace_sos = replace_sos

        self.logzero = -10000000000.0

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_prev, c_prev):
        z_list = [None] * self.dlayers
        c_list = None
        if self.dtype == "lstm":
            c_list = [None] * self.dlayers
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l]))
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for l in six.moves.range(1, self.dlayers):
                z_list[l] = self.decoder[l](self.dropout_dec[l - 1](z_list[l - 1]), z_prev[l])
        return z_list, c_list

    def forward(self, hs_pad, hlens, ys_pad, strm_idx=0, lang_ids=None):
        """Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
                                    [in multi-encoder case,
                                    list of torch.Tensor, [(B, Tmax_1, D), (B, Tmax_2, D), ..., ] ]
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
                                    [in multi-encoder case, list of torch.Tensor, [(B), (B), ..., ]
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :param int strm_idx: stream index indicates the index of decoding stream.
        :param torch.Tensor lang_ids: batch of target language id tensor (B, 1)
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        """
        # to support mutiple encoder asr mode, in single encoder mode, convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            hs_pad = [hs_pad]
            hlens = [hlens]

        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        # attention index for the attention module
        # in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlens should be list of list of integer
        hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)
        for idx in range(self.num_encs):
            logging.info(
                self.__class__.__name__ + 'Number of Encoder:{}; enc{}: input lengths: {}.'.format(self.num_encs,
                                                                                                   idx + 1, hlens[idx]))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # initialization
        c_list = [self.zero_state(hs_pad[0])]
        z_list = [self.zero_state(hs_pad[0])]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad[0]))
            z_list.append(self.zero_state(hs_pad[0]))
        z_all = []
        if self.num_encs == 1:
            att_w = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            if self.num_encs == 1:
                att_c, att_w = self.att[att_idx](hs_pad[0], hlens[0], self.dropout_dec[0](z_list[0]), att_w)
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att[idx](hs_pad[idx], hlens[idx],
                                                                     self.dropout_dec[0](z_list[0]), att_w_list[idx])
                hs_pad_han = torch.stack(att_c_list, dim=1)
                hlens_han = [self.num_encs] * len(ys_in)
                att_c, att_w_list[self.num_encs] = self.att[self.num_encs](hs_pad_han, hlens_han,
                                                                           self.dropout_dec[0](z_list[0]),
                                                                           att_w_list[self.num_encs])
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach().cpu(), axis=1)
                z_out = self.dropout_emb(self.embed(to_device(self, z_out)))
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list)
            if self.context_residual:
                z_all.append(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))  # utt x (zdim + hdim)
            else:
                z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)

        z_all = torch.stack(z_all, dim=1).view(batch * olength, -1)
        # compute loss
        y_all = self.output(z_all)
        if LooseVersion(torch.__version__) < LooseVersion('1.0'):
            reduction_str = 'elementwise_mean'
        else:
            reduction_str = 'mean'
        self.loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
                                    ignore_index=self.ignore_id,
                                    reduction=reduction_str)
        # compute perplexity
        ppl = math.exp(self.loss.item())
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = th_accuracy(y_all, ys_out_pad, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.item()).split('\n')))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = y_all.view(batch, olength, -1)
            ys_true = ys_out_pad
            for (i, y_hat), y_true in zip(enumerate(ys_hat.detach().cpu().numpy()),
                                          ys_true.detach().cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_device(self, torch.from_numpy(self.labeldist))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc, ppl

    def initial_decoding_state(self, h, rnnlm, strm_idx=0):
        # This is per hypotesis state
        att_idx = min(strm_idx, len(self.att) - 1)
        state = {'att_idx': att_idx, 'c_prev': None}

        if rnnlm:
            state['rnnlm_prev'] = None

        if self.dtype == 'lstm':
            state['c_prev'] = [self.zero_state(h[0].unsqueeze(0)) for _ in range(self.dlayers)]
        state['z_prev'] = [self.zero_state(h[0].unsqueeze(0)) for _ in range(self.dlayers)]

        if self.num_encs == 1:
            a = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs + 1)  # atts + han
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han
        state['a_prev'] = a
        return state

    def decode_from_state(self, state, h, vy):
        new_state = {'att_idx': state['att_idx'], 'c_prev': None}

        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
        if self.num_encs == 1:
            att_c, att_w = self.att[state['att_idx']](
                h[0].unsqueeze(0),
                [h[0].size(0)],
                self.dropout_dec[0](state['z_prev'][0]),
                state['a_prev']
            )
        else:
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs):
                att_c_list[idx], att_w_list[idx] = self.att[idx](
                    h[idx].unsqueeze(0),
                    [h[idx].size(0)],
                    self.dropout_dec[0](state['z_prev'][0]),
                    state['a_prev'][idx]
                )
            h_han = torch.stack(att_c_list, dim=1)
            att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                h_han,
                [self.num_encs],
                self.dropout_dec[0](state['z_prev'][0]),
                state['a_prev'][self.num_encs]
            )
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = self.rnn_forward(ey, state['z_prev'], state['c_prev'])

        if self.num_encs == 1:
            new_state['a_prev'] = att_w
        else:
            new_state['a_prev'] = att_w_list
        new_state['z_prev'] = z_list
        if self.dtype == 'lstm':
            new_state['c_prev'] = c_list

        # get nbest local scores and their ids
        if self.context_residual:
            logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        local_att_scores = F.log_softmax(logits, dim=1)
        return new_state, local_att_scores

    def encode_for_beam(self, h):
        """Put encoded data in the correct shape."""
        if self.num_encs == 1:
            return [h]
        maxlen = np.amin([h[idx].size(0) for idx in range(self.num_encs)])
        return h, maxlen

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None, strm_idx=0):
        """beam search implementation

        :param torch.Tensor h: encoder hidden state (T, eprojs)
                                [in multi-encoder case, list of torch.Tensor, [(T1, eprojs), (T2, eprojs), ...] ]
        :param torch.Tensor lpz: ctc log softmax output (T, odim)
                                [in multi-encoder case, list of torch.Tensor, [(T1, odim), (T2, odim), ...] ]
        :param Namespace recog_args: argument Namespace containing options
        :param char_list: list of character strings
        :param torch.nn.Module rnnlm: language module
        :param int strm_idx: stream index for speaker parallel attention in multi-speaker case
        :return: N-best decoding results
        :rtype: list of dicts
        """
        model = self
        bs = BeamSearch(model, recog_args, char_list, self.replace_sos)
        nbest_hyp = bs.recognize_beam(h, lpz, rnnlm=rnnlm, strm_idx=strm_idx)
        return nbest_hyp

    def encode_for_beam_batch(self, h, hlens):
        """Put encoded data in the correct shape."""
        if self.num_encs == 1:
            h = [h]
            hlens = [hlens]

        for idx in range(self.num_encs):
            h[idx] = mask_by_length(h[idx], hlens[idx], 0.0)
        return h, hlens

    def initial_decoding_state_batch(self, h, strm_idx, batch_size, beam_size):
        # h is not needed here but it might for other models
        att_idx = min(strm_idx, len(self.att) - 1)
        state = {'att_idx':att_idx}

        max_n_hyps = batch_size*beam_size
        c_prev = None
        if self.dtype == 'lstm':
            c_prev = [to_device(self, torch.zeros(max_n_hyps, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_device(self, torch.zeros(max_n_hyps, self.dunits)) for _ in range(self.dlayers)]
        state['z_prev'] = z_prev
        state['c_prev'] = c_prev

        if self.num_encs == 1:
            a_prev = [None]
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a_prev = [None] * (self.num_encs + 1)  # atts + han
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han
        state['a_prev'] = a_prev


        h, hlens = h
        exp_hlens = [
            hlen.repeat(beam_size).view(beam_size, batch_size).transpose(0, 1).reshape(-1).tolist()
            for hlen in hlens
        ]
        exp_h = [
            hi.unsqueeze(1).repeat(1, beam_size, 1, 1).reshape(n_bb, *hi.size()[1:3])
            for hi in h
        ]
        state['exp_hlens'] = exp_hlens
        state['exp_h'] = exp_h
        return state

    def decode_from_state_batch(self, state, h, vy, batch_size, beam_size, vidx):
        n_bb = batch_size * beam_size

        if vidx is not None:
            att_w_list = state['att_w_list']
            a_prev = []
            num_atts = 1 if self.num_encs == 1 else self.num_encs + 1
            for idx in range(num_atts):
                if isinstance(att_w_list[idx], torch.Tensor):
                    _a_prev = torch.index_select(att_w_list[idx].view(n_bb, *att_w_list[idx].shape[1:]), 0, vidx)
                elif isinstance(att_w_list[idx], list):
                    # handle the case of multi-head attention
                    _a_prev = [torch.index_select(att_w_one.view(n_bb, -1), 0, vidx) for att_w_one in att_w_list[idx]]
                else:
                    # handle the case of location_recurrent when return is a tuple
                    _a_prev_ = torch.index_select(att_w_list[idx][0].view(n_bb, -1), 0, vidx)
                    _h_prev_ = torch.index_select(att_w_list[idx][1][0].view(n_bb, -1), 0, vidx)
                    _c_prev_ = torch.index_select(att_w_list[idx][1][1].view(n_bb, -1), 0, vidx)
                    _a_prev = (_a_prev_, (_h_prev_, _c_prev_))
                a_prev.append(_a_prev)
            z_prev = [torch.index_select(state['z_prev'][li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]
            if self.dtype == 'lstm':
                c_prev = [torch.index_select(state['c_list'][li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]

        ey = self.dropout_emb(self.embed(vy))
        if self.num_encs == 1:
            att_c, att_w = self.att[state['att_idx']](state['exp_h'][0], state['exp_hlens'][0], self.dropout_dec[0](z_prev[0]), a_prev[0])
            att_w_list = [att_w]
        else:
            att_c_list = [None] * self.num_encs
            att_w_list = [None] * (self.num_encs + 1)
            for idx in range(self.num_encs):
                att_c_list[idx], att_w_list[idx] = self.att[idx](
                    state['exp_h'][idx],
                    state['exp_hlens'][idx],
                    self.dropout_dec[0](z_prev[0]), a_prev[idx]
                )
            exp_h_han = torch.stack(att_c_list, dim=1)
            att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                exp_h_han, [self.num_encs] * n_bb, self.dropout_dec[0](z_prev[0]), a_prev[self.num_encs]
            )
        state['att_w_list'] = att_w_list
        ey = torch.cat((ey, att_c), dim=1)

        z_list, c_list = self.rnn_forward(ey, z_prev, c_prev)
        state['z_prev'] = z_list
        state['c_prev'] = c_list

        if self.context_residual:
            logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        local_att_scores = F.log_softmax(logits, dim=1)

        # TODO update return type in signature (doc)
        # attention list needed for ctc
        return state, att_w_list, local_att_scores

    def recognize_beam_batch(self, h, hlens, lpz, recog_args, char_list, rnnlm=None,
                             normalize_score=True, strm_idx=0, lang_ids=None):
        # to support mutiple encoder asr mode, in single encoder mode, convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            h = [h]
            hlens = [hlens]
            lpz = [lpz]
        if self.num_encs > 1 and lpz is None:
            lpz = [lpz] * self.num_encs

        att_idx = min(strm_idx, len(self.att) - 1)
        for idx in range(self.num_encs):
            logging.info(
                'Number of Encoder:{}; enc{}: input lengths: {}.'.format(self.num_encs, idx + 1, h[idx].size(1)))
            h[idx] = mask_by_length(h[idx], hlens[idx], 0.0)

        # search params
        batch = len(hlens[0])
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = getattr(recog_args, "ctc_weight", 0)  # for NMT
        att_weight = 1.0 - ctc_weight
        ctc_margin = getattr(recog_args, "ctc_window_margin", 0)  # use getattr to keep compatibility
        # weights-ctc, e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss
        if lpz[0] is not None and self.num_encs > 1:
            weights_ctc_dec = recog_args.weights_ctc_dec / np.sum(recog_args.weights_ctc_dec)  # normalize
            logging.info('ctc weights (decoding): ' + ' '.join([str(x) for x in weights_ctc_dec]))
        else:
            weights_ctc_dec = [1.0]

        n_bb = batch * beam
        pad_b = to_device(self, torch.arange(batch) * beam).view(-1, 1)

        max_hlen = np.amin([max(hlens[idx]) for idx in range(self.num_encs)])
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialization
        c_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        vscores = to_device(self, torch.zeros(batch, beam))

        rnnlm_state = None
        if self.num_encs == 1:
            a_prev = [None]
            att_w_list, ctc_scorer, ctc_state = [None], [None], [None]
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a_prev = [None] * (self.num_encs + 1)  # atts + han
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            ctc_scorer, ctc_state = [None] * (self.num_encs), [None] * (self.num_encs)
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        if self.replace_sos and recog_args.tgt_lang:
            logging.info('<sos> index: ' + str(char_list.index(recog_args.tgt_lang)))
            logging.info('<sos> mark: ' + recog_args.tgt_lang)
            yseq = [[char_list.index(recog_args.tgt_lang)] for _ in six.moves.range(n_bb)]
        elif lang_ids is not None:
            # NOTE: used for evaluation during training
            yseq = [[lang_ids[b // recog_args.beam_size]] for b in six.moves.range(n_bb)]
        else:
            logging.info('<sos> index: ' + str(self.sos))
            logging.info('<sos> mark: ' + char_list[self.sos])
            yseq = [[self.sos] for _ in six.moves.range(n_bb)]

        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = [hlens[idx].repeat(beam).view(beam, batch).transpose(0, 1).contiguous() for idx in
                     range(self.num_encs)]
        exp_hlens = [exp_hlens[idx].view(-1).tolist() for idx in range(self.num_encs)]
        exp_h = [h[idx].unsqueeze(1).repeat(1, beam, 1, 1).contiguous() for idx in range(self.num_encs)]
        exp_h = [exp_h[idx].view(n_bb, h[idx].size()[1], h[idx].size()[2]) for idx in range(self.num_encs)]

        if lpz[0] is not None:
            scoring_ratio = CTC_SCORING_RATIO if att_weight > 0.0 and not lpz[0].is_cuda else 0
            ctc_scorer = [CTCPrefixScoreTH(lpz[idx], hlens[idx], 0, self.eos, beam,
                                           scoring_ratio, margin=ctc_margin) for idx in range(self.num_encs)]

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            vy = to_device(self, torch.LongTensor(self._get_last_yseq(yseq)))
            ey = self.dropout_emb(self.embed(vy))
            if self.num_encs == 1:
                att_c, att_w = self.att[att_idx](exp_h[0], exp_hlens[0], self.dropout_dec[0](z_prev[0]), a_prev[0])
                att_w_list = [att_w]
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att[idx](exp_h[idx], exp_hlens[idx],
                                                                     self.dropout_dec[0](z_prev[0]), a_prev[idx])
                exp_h_han = torch.stack(att_c_list, dim=1)
                att_c, att_w_list[self.num_encs] = self.att[self.num_encs](exp_h_han, [self.num_encs] * n_bb,
                                                                           self.dropout_dec[0](z_prev[0]),
                                                                           a_prev[self.num_encs])
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list, c_list = self.rnn_forward(ey, z_prev, c_prev)
            if self.context_residual:
                logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
            else:
                logits = self.output(self.dropout_dec[-1](z_list[-1]))
            local_scores = att_weight * F.log_softmax(logits, dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores

            # ctc
            if ctc_scorer[0]:
                for idx in range(self.num_encs):
                    att_w = att_w_list[idx]
                    att_w_ = att_w if isinstance(att_w, torch.Tensor) else att_w[0]
                    ctc_state[idx], local_ctc_scores = ctc_scorer[idx](yseq, ctc_state[idx], local_scores, att_w_)
                    local_scores = local_scores + ctc_weight * weights_ctc_dec[idx] * local_ctc_scores

            local_scores = local_scores.view(batch, beam, self.odim)
            if i == 0:
                local_scores[:, 1:, :] = self.logzero

            # accumulate scores
            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, -1)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
            accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            accum_padded_beam_ids = (torch.div(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist()

            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_device(self, torch.LongTensor(accum_padded_beam_ids))

            a_prev = []
            num_atts = self.num_encs if self.num_encs == 1 else self.num_encs + 1
            for idx in range(num_atts):
                if isinstance(att_w_list[idx], torch.Tensor):
                    _a_prev = torch.index_select(att_w_list[idx].view(n_bb, *att_w_list[idx].shape[1:]), 0, vidx)
                elif isinstance(att_w_list[idx], list):
                    # handle the case of multi-head attention
                    _a_prev = [torch.index_select(att_w_one.view(n_bb, -1), 0, vidx) for att_w_one in att_w_list[idx]]
                else:
                    # handle the case of location_recurrent when return is a tuple
                    _a_prev_ = torch.index_select(att_w_list[idx][0].view(n_bb, -1), 0, vidx)
                    _h_prev_ = torch.index_select(att_w_list[idx][1][0].view(n_bb, -1), 0, vidx)
                    _c_prev_ = torch.index_select(att_w_list[idx][1][1].view(n_bb, -1), 0, vidx)
                    _a_prev = (_a_prev_, (_h_prev_, _c_prev_))
                a_prev.append(_a_prev)
            z_prev = [torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]
            if self.dtype == 'lstm':
                c_prev = [torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]

            # pick ended hyps
            if i >= minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        _vscore = None
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            if len(yk) <= min(hlens[idx][samp_i] for idx in range(self.num_encs)):
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                        elif i == maxlen - 1:
                            yk = yseq[k][:]
                            _vscore = vscores[samp_i][beam_j] + penalty_i
                        if _vscore:
                            yk.append(self.eos)
                            if rnnlm:
                                _vscore += recog_args.lm_weight * rnnlm.final(rnnlm_state, index=k)
                            _score = _vscore.data.cpu().numpy()
                            ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score})
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            if rnnlm:
                rnnlm_state = self._index_select_lm_state(rnnlm_state, 0, vidx)
            if ctc_scorer[0]:
                for idx in range(self.num_encs):
                    ctc_state[idx] = ctc_scorer[idx].index_select_state(ctc_state[idx], accum_best_ids)

        torch.cuda.empty_cache()

        dummy_hyps = [{'yseq': [self.sos, self.eos], 'score': np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        if normalize_score:
            for samp_i in six.moves.range(batch):
                for x in ended_hyps[samp_i]:
                    x['score'] /= len(x['yseq'])

        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in six.moves.range(batch)]

        return nbest_hyps

    def calculate_all_attentions(self, hs_pad, hlen, ys_pad, strm_idx=0, lang_ids=None):
        """Calculate all of attentions

            :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
                                        [in multi-encoder case,
                                        list of torch.Tensor, [(B, Tmax_1, D), (B, Tmax_2, D), ..., ] ]
            :param torch.Tensor hlen: batch of lengths of hidden state sequences (B)
                                        [in multi-encoder case, list of torch.Tensor, [(B), (B), ..., ]
            :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
            :param int strm_idx: stream index for parallel speaker attention in multi-speaker case
            :param torch.Tensor lang_ids: batch of target language id tensor (B, 1)
            :return: attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) multi-encoder case => [(B, Lmax, Tmax1), (B, Lmax, Tmax2), ..., (B, Lmax, NumEncs)]
                3) other case => attention weights (B, Lmax, Tmax).
            :rtype: float ndarray
        """
        # to support mutiple encoder asr mode, in single encoder mode, convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            hs_pad = [hs_pad]
            hlen = [hlen]

        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlen should be list of list of integer
        hlen = [list(map(int, hlen[idx])) for idx in range(self.num_encs)]

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = ys_out_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad[0])]
        z_list = [self.zero_state(hs_pad[0])]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad[0]))
            z_list.append(self.zero_state(hs_pad[0]))
        att_ws = []
        if self.num_encs == 1:
            att_w = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            if self.num_encs == 1:
                att_c, att_w = self.att[att_idx](hs_pad[0], hlen[0], self.dropout_dec[0](z_list[0]), att_w)
                att_ws.append(att_w)
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att[idx](hs_pad[idx], hlen[idx],
                                                                     self.dropout_dec[0](z_list[0]), att_w_list[idx])
                hs_pad_han = torch.stack(att_c_list, dim=1)
                hlen_han = [self.num_encs] * len(ys_in)
                att_c, att_w_list[self.num_encs] = self.att[self.num_encs](hs_pad_han, hlen_han,
                                                                           self.dropout_dec[0](z_list[0]),
                                                                           att_w_list[self.num_encs])
                att_ws.append(att_w_list)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list)

        if self.num_encs == 1:
            # convert to numpy array with the shape (B, Lmax, Tmax)
            att_ws = att_to_numpy(att_ws, self.att[att_idx])
        else:
            _att_ws = []
            for idx, ws in enumerate(zip(*att_ws)):
                ws = att_to_numpy(ws, self.att[idx])
                _att_ws.append(ws)
            att_ws = _att_ws
        return att_ws

    @staticmethod
    def _get_last_yseq(exp_yseq):
        last = []
        for y_seq in exp_yseq:
            last.append(y_seq[-1])
        return last

    @staticmethod
    def _append_ids(yseq, ids):
        if isinstance(ids, list):
            for i, j in enumerate(ids):
                yseq[i].append(j)
        else:
            for i in range(len(yseq)):
                yseq[i].append(ids)
        return yseq

    @staticmethod
    def _index_select_list(yseq, lst):
        new_yseq = []
        for l in lst:
            new_yseq.append(yseq[l][:])
        return new_yseq

    @staticmethod
    def _index_select_lm_state(rnnlm_state, dim, vidx):
        if isinstance(rnnlm_state, dict):
            new_state = {
                k:[torch.index_select(vi, dim, vidx) for vi in v]
                for k, v in rnnlm_state.items()
            }
        elif isinstance(rnnlm_state, list):
            new_state = [rnnlm_state[int(i)][:] for i in vidx]
        return new_state

    # scorer interface methods
    def init_state(self, x):
        # to support mutiple encoder asr mode, in single encoder mode, convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        c_list = [self.zero_state(x[0].unsqueeze(0))]
        z_list = [self.zero_state(x[0].unsqueeze(0))]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(x[0].unsqueeze(0)))
            z_list.append(self.zero_state(x[0].unsqueeze(0)))
        # TODO(karita): support strm_index for `asr_mix`
        strm_index = 0
        att_idx = min(strm_index, len(self.att) - 1)
        if self.num_encs == 1:
            a = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs + 1)  # atts + han
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han
        return dict(c_prev=c_list[:], z_prev=z_list[:], a_prev=a, workspace=(att_idx, z_list, c_list))

    def score(self, yseq, state, x):
        # to support mutiple encoder asr mode, in single encoder mode, convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        att_idx, z_list, c_list = state["workspace"]
        vy = yseq[-1].unsqueeze(0)
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
        if self.num_encs == 1:
            att_c, att_w = self.att[att_idx](
                x[0].unsqueeze(0), [x[0].size(0)],
                self.dropout_dec[0](state['z_prev'][0]), state['a_prev'])
        else:
            att_w = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs):
                att_c_list[idx], att_w[idx] = self.att[idx](x[idx].unsqueeze(0), [x[idx].size(0)],
                                                            self.dropout_dec[0](state['z_prev'][0]),
                                                            state['a_prev'][idx])
            h_han = torch.stack(att_c_list, dim=1)
            att_c, att_w[self.num_encs] = self.att[self.num_encs](h_han, [self.num_encs],
                                                                  self.dropout_dec[0](state['z_prev'][0]),
                                                                  state['a_prev'][self.num_encs])
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = self.rnn_forward(ey, state['z_prev'], state['c_prev'])
        if self.context_residual:
            logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        logp = F.log_softmax(logits, dim=1).squeeze(0)
        return logp, dict(c_prev=c_list, z_prev=z_list, a_prev=att_w, workspace=(att_idx, z_list, c_list))


def decoder_for(args, odim, sos, eos, att, labeldist):
    return Decoder(args.eprojs, odim, args.dtype, args.dlayers, args.dunits, sos, eos, att, args.verbose,
                   args.char_list, labeldist,
                   args.lsm_weight, args.sampling_probability, args.dropout_rate_decoder,
                   getattr(args, "context_residual", False),  # use getattr to keep compatibility
                   getattr(args, "replace_sos", False),  # use getattr to keep compatibility
                   getattr(args, "num_encs", 1))  # use getattr to keep compatibility

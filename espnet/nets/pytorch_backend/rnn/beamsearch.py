"""BeamSearch module."""

import logging
import numpy as np
import six
import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect

CTC_SCORING_RATIO = 1.5


def find_first_tensor(list_of_tensor_or_tensor):
    """Find the first tensor in a tree."""
    t = type(list_of_tensor_or_tensor)
    if t is torch.Tensor:
        return list_of_tensor_or_tensor
    if t is list and len(t):
        return find_first_tensor(list_of_tensor_or_tensor[0])
    return torch.Tensor()


def find_all_tensors(l):
    """Find all the tensors in a tree."""
    t = type(l)
    tensors = []
    if t is list:
        for sub_l in l:
            tensors.extend(find_all_tensors(sub_l))
    if t is torch.Tensor:
        tensors.append(l)
    return tensors


class Hypothesis(dict):
    """Hypothesis class.

    :param any model_state: stored state given by the model
    """

    def __init__(self, model_state=None):
        """Construct an Hypothesis object.

        :param any model_state: stored state given by the model
        """
        super(Hypothesis).__init__()
        self.model_state = model_state


class BeamableModel:
    """BeamableModel class.

    :param any model_state: stored state given by the model
    """

    def encode_for_beam(self, x):
        """Return the projection h of x to start decoding."""
        raise NotImplementedError

    def initial_decoding_state(self):
        """Give the state to start a new decoding."""
        raise NotImplementedError

    def decode_from_state(self, state, h, vy):
        """Advance one step the decoding of h from state, vy is the last decoded element."""
        # and returns its decoding state
        raise NotImplementedError


class BeamSearch:
    """BeamSearch module.

    :param model BeamableModel
    :param recog_args: program arguments
    :param char_list
    :param replace_sos

    """

    def __init__(self, model, recog_args, char_list, replace_sos):
        """Construct a BeacmSearch Object."""
        self.recog_args = recog_args
        self.char_list = char_list
        self.model = model

        # search params
        self.beam_size = recog_args.beam_size
        self.penalty = recog_args.penalty
        self.ctc_weight = getattr(recog_args, "ctc_weight", False)  # for NMT

        # start of sequence
        if replace_sos and self.recog_args.tgt_lang:
            self.sos = char_list.index(recog_args.tgt_lang)
        else:
            self.sos = model.sos

    def recognize_beam(self, x, lpz, rnnlm=None, minlenratio=None):
        """Return the nbest hypotheses for x."""
        # when we retry with a different minlenratio,
        # no need to duplicate recog_args
        if minlenratio is None:
            minlenratio = self.recog_args.minlenratio

        h = self.model.encode_for_beam(x)
        # vy is a tensor containing the id
        # of the last decoded element at each step
        vy = find_first_tensor(h).new_zeros(1).long()

        # maxlen minlen calculation
        maxlen = np.amin([t.size(0) for t in find_all_tensors(h)])
        if self.recog_args.maxlenratio != 0:
            maxlen = int(self.recog_args.maxlenratio * maxlen)
        maxlen = max(1, maxlen)  # maxlen >= 1
        minlen = int(minlenratio * maxlen)

        # initialize hypothesis
        hyp = Hypothesis(self.model.initial_decoding_state(h))
        hyps = [hyp]

        hyp['score'] = 0.0
        hyp['yseq'] = [self.sos]
        if rnnlm:
            hyp['rnnlm_prev'] = None

        # initialize ctc
        has_ctc = (lpz is not None)
        if has_ctc:

            if type(lpz) is not list:
                lpz = [lpz]
            num_ctc = len(lpz)

            if num_ctc > 1:
                weights_ctc_dec = (
                    self.recog_args.weights_ctc_dec /
                    np.sum(self.recog_args.weights_ctc_dec)
                )
            else:
                weights_ctc_dec = [1.0]

            ctc_prefix_score = [
                CTCPrefixScore(lpz[idx].detach().numpy(), 0, self.model.eos, np)
                for idx in range(num_ctc)
            ]
            hyp['ctc_state_prev'] = [
                ctc_prefix_score[idx].initial_state()
                for idx in range(num_ctc)
            ]
            hyp['ctc_score_prev'] = [0.0] * num_ctc

            ctc_beam_size = lpz[0].shape[-1]
            if self.ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam_size = min(
                    ctc_beam_size,
                    int(self.beam_size * CTC_SCORING_RATIO)
                )

        # ┌────────────────────────────────────────────────────────┐
        # │ look for the nbest hyps at each time step by advancing │
        # │ all current hypotheses by one step                     │
        # └────────────────────────────────────────────────────────┘
        ended_hyps = []
        for i in six.moves.range(maxlen):
            hyps_best_kept = []
            # for each hypothesis
            for hyp in hyps:
                # step forward the decoding
                vy[0] = hyp['yseq'][i]
                model_state, local_att_scores = self.model.decode_from_state(
                    hyp.model_state, h, vy
                )
                # local_att_scores = F.log_softmax(logits, dim=1)

                # ┌─────────────────────────────────────────────────────────┐
                # │ prepare the generation of the next beam_size hypotheses │
                # └─────────────────────────────────────────────────────────┘
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(
                        hyp['rnnlm_prev'], vy
                    )
                    local_scores = (
                        local_att_scores +
                        self.recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if has_ctc:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam_size, dim=1
                    )
                    local_scores = (1.0 - self.ctc_weight) * local_best_scores

                    ctc_scores = [None] * num_ctc
                    ctc_states = [None] * num_ctc
                    for idx in range(num_ctc):
                        ctc_scores[idx], ctc_states[idx] = (
                            ctc_prefix_score[idx](
                                hyp['yseq'],
                                local_best_ids[0],
                                hyp['ctc_state_prev'][idx]
                            )
                        )
                        local_scores += (
                            self.ctc_weight * weights_ctc_dec[idx] *
                            torch.from_numpy(
                                ctc_scores[idx] - hyp['ctc_score_prev'][idx]
                            )
                        )

                    if rnnlm:
                        local_scores += (
                            self.recog_args.lm_weight *
                            local_lm_scores[:, local_best_ids[0]]
                        )

                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, self.beam_size, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, self.beam_size, dim=1
                    )

                # ┌─────────────────────────────────────────┐
                # │ genereate the next beam_size hypotheses │
                # └─────────────────────────────────────────┘
                for j in six.moves.range(self.beam_size):
                    new_hyp = Hypothesis(model_state)
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = hyp['yseq'][:].append(
                        int(local_best_ids[0, j])
                    )

                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state

                    if has_ctc:
                        new_hyp['ctc_state_prev'] = [
                            ctc_states[idx][joint_best_ids[0, j]]
                            for idx in range(self.num_encs)
                        ]
                        new_hyp['ctc_score_prev'] = [
                            ctc_scores[idx][joint_best_ids[0, j]]
                            for idx in range(self.num_encs)
                        ]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True
                )[:self.beam_size]

            # ┌─────────────────────────────────────┐
            # │ treat the newly explored hypotheses │
            # └─────────────────────────────────────┘
            # sort and get nbest
            hyps = hyps_best_kept

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last position in the loop')
            if i == maxlen - 1:
                logging.info('adding <eos> in the last position in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypotheses to a final list,
            # and remove them from current hypotheses
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    penalty = (i + 1) * self.penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += (
                                self.recog_args.lm_weight *
                                rnnlm.final(hyp['rnnlm_prev'])
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and self.recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if not len(hyps):
                break
        # ┌────────────────────────────────────────────────────────┐
        # │ choose between all the ended Hypotheses the nbest ones │
        # └────────────────────────────────────────────────────────┘
        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True
        )[:min(len(ended_hyps), self.recog_args.nbest)]

        # check number of hypotheses
        if len(nbest_hyps) == 0 and minlenratio != 0:
            logging.warning(
                'there is no N-best results, '
                'perform recognition again with smaller minlenratio.'
            )
            minlenratio = max(0.0, self.recog_args.minlenratio - 0.1)
            return self.recognize_beam(
                x, lpz, rnnlm, minlenratio=minlenratio
            )
        return nbest_hyps

    def recognize_beam_batch(self, xs):
        """Return the nbest Hypothesis for each element of the batch xs."""
        raise NotImplementedError

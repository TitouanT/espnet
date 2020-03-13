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
    if t is list and len(list_of_tensor_or_tensor):
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

    Args:
        model_state (any): stored state given by the model
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

    # ┌─────────────────┐
    # │ for decode_beam │
    # └─────────────────┘
    def encode_for_beam(self, x):
        """
        Return the projection h of x to start decoding.

        Args:
            x (ndarray): list of input features

        Returns:
            Tuple(
                h (Torch.tensor): hidden, non recurrent, representation of the batch by the model
                maxlen (int): the maximum length of an output sequence
            )
        """
        raise NotImplementedError

    def initial_decoding_state(self, h, strm_idx):
        """
        Give the initial recurrent state to start a new decoding.

        Args:
            h (Torch.tensor): hidden, non recurrent, representation of the input by the model
            strm_idx (int): stream index indicates the index of decoding stream.

        Returns:
            state: initial recurrent state to start a new decoding.
        """
        raise NotImplementedError

    def decode_from_state(self, state, h, vy):
        """
        Advance one step the decoding of h from state, vy is the last decoded element.

        Args:
            state (any): the previous state that generated vy
            h (Torch.tensor): the encoded input
            vy (int): id of the last decoded element.

        Returns:
            state (any): current recurrent state of the model
        """
        # and returns its decoding state
        raise NotImplementedError

    # ┌───────────────────────┐
    # │ for decode_beam_batch │
    # └───────────────────────┘
    def initial_decoding_state_batch(self, hs, strm_idx):
        """
        Give the initial recurrent state of the model to start a new batch decoding.

        Args:
            hs (Torch.tensor): hidden, non recurrent, representation of the batch by the model
            strm_idx (int): stream index indicates the index of decoding stream.

        Returns:
            state: initial recurrent state to start a new batch decoding.
        """
        raise NotImplementedError

    def encode_for_beam_batch(self, xs):
        """
        Return the projections h for each x in xs to start decoding.

        Args:
            xs (ndarray): list of input

        Returns:
            # Tuple(
                hs (Torch.tensor): hidden, non recurrent, representation of the batch by the model
                # maxlen (int): the maximum length of an output sequence
            # )
        """
        raise NotImplementedError

    def decode_from_state_batch(self, state, hs, vys, batch_size, beam_size, vidx):
        """
        Advance one step the decoding of hs from state, vys are the last decoded elements.

        Args:
            state (any): the previous state that generated vys
            hs (Torch.tensor): the encoded batch
            vys (list): list of the decoded elements at the last step
            batch_size (int)
            beam_size (int)
            vidx (torch.LongTensor): TODO

        Returns:
            state (any): current recurrent state of the model
            att_w_list (list): list of attention test, needed for ctc optimization
            local_att_scores (Torch.tensor): ouput for this step
        """
        # and returns its decoding state
        raise NotImplementedError


class BeamSearch:
    """BeamSearch module.

    Args:
        model (BeamableModel): encode and decode the input
        recog_args     (Dict): the arguments of the program
        char_list      (list): sequence output vocabulary
        replace_sos    (Bool): use for multilingual (speech/text) translation
    """

    def __init__(self, model, recog_args, char_list, replace_sos):
        """Construct a BeacmSearch Object."""
        self.recog_args = recog_args
        self.char_list = char_list
        self.model = model
        self.minlenratio = model.minlenratio
        self.odim = self.model.odim
        self.logzero = self.model.logzero

        # search params
        self.beam_size = recog_args.beam_size
        self.penalty = recog_args.penalty
        self.ctc_weight = getattr(recog_args, "ctc_weight", 0)  # for NMT
        self.att_weight = 1 - self.ctc_weight
        self.ctc_margin = getattr(recog_args, "ctc_window_margin", 0)  # use getattr to keep compatibility
        self.eos = model.eos

        # start of sequence
        if self.replace_sos and recog_args.tgt_lang:
            self.sos = char_list.index(recog_args.tgt_lang)
        else:
            self.sos = model.sos
        self.eos = self.model.eos

    def recognize_beam(self, x, lpz, rnnlm=None, minlenratio=None, strm_idx=0):
        """Return the nbest hypotheses for x."""
        # when we retry with a different minlenratio,
        # no need to duplicate recog_args
        if minlenratio is None:
            minlenratio = self.minlenratio

        h, maxlen = self.model.encode_for_beam(x)
        # vy is a tensor containing the id
        # of the last decoded element at each step
        vy = find_first_tensor(h).new_zeros(1).long()

        # maxlen minlen calculation
        if self.recog_args.maxlenratio != 0:
            maxlen = int(self.recog_args.maxlenratio * maxlen)
        maxlen = max(1, maxlen)  # maxlen >= 1
        minlen = int(minlenratio * maxlen)

        # initialize hypothesis
        hyp = Hypothesis(self.model.initial_decoding_state(h, rnnlm, lpz, strm_idx))
        hyps = [hyp]

        hyp['score'] = 0.0
        hyp['yseq'] = [self.sos]

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
                CTCPrefixScore(lpz[idx].detach().numpy(), 0, self.eos, np)
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
                    local_scores = self.att_weight * local_best_scores

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
                    new_hyp['yseq'] = hyp['yseq'][:]
                    new_hyp['yseq'].append(int(local_best_ids[0, j]))

                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state

                    if has_ctc:
                        new_hyp['ctc_state_prev'] = [
                            ctc_states[idx][joint_best_ids[0, j]]
                            for idx in range(num_ctc)
                        ]
                        new_hyp['ctc_score_prev'] = [
                            ctc_scores[idx][joint_best_ids[0, j]]
                            for idx in range(num_ctc)
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
            minlenratio = max(0.0, minlenratio - 0.1)
            return self.recognize_beam(
                x, lpz, rnnlm, minlenratio=minlenratio
            )
        return nbest_hyps

    def recognize_beam_batch(
        self, xs, lpz, rnnlm=None, normalize_score=True, strm_idx=0, lang_ids=None, minlenratio=None
    ):
        """Return the nbest Hypothesis for each element of the batch xs."""
        batch_size = len(xs)
        if minlenratio is None:
            minlenratio = self.minlenratio
        h, hlens = self.model.encode_for_beam_batch(x)  # first dimension is the number of encoder ? might be an issue, might not, will see

        maxlen = np.amin([max(hlen) for hlen in hlens])
        if self.recog_args.maxlenratio != 0:
            maxlen = int(self.recog_args.maxlenratio * maxlen)
        maxlen = max(1, maxlen)  # maxlen >= 1
        minlen = int(minlenratio * maxlen)

        has_ctc = (lpz is not None)
        if has_ctc:
            if type(lpz) is not list:
                lpz = [lpz]
            num_ctc = len(lpz)
            assert(num_ctc == len(hlens))

            # weights-ctc, e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss
            if num_ctc > 1:
                weights_ctc_dec = (
                    self.recog_args.weights_ctc_dec /
                    np.sum(self.recog_args.weights_ctc_dec)
                )
            else:
                weights_ctc_dec = [1.0]

            ctc_state = [None] * num_ctc
            ctc_scorer = [
                CTCPrefixScoreTH(
                    lpz[idx], hlens[idx], 0, self.eos, self.beam_size,
                    CTC_SCORING_RATIO if self.att_weight > 0.0 and not lpz[0].is_cuda else 0,
                    margin=ctc_margin
                )
                for idx in range(num_ctc)
            ]

        pad_b = to_device(self.model, torch.arange(batch_size) * self.beam_size).view(-1, 1) # state ?
        n_bb = batch_size * self.beam_size  # n_bb

        vscores = to_device(self, torch.zeros(batch_size, self.beam_size))
        model_state = self.model.initial_decoding_state_batch(h, strm_idx, n_bb)

        rnnlm_state = None

        if lang_ids is not None:
            yseq = [[lang_ids[b // recog_args.self.beam_size]] for b in six.moves.range(n_bb)]
        else:
            yseq = [[self.sos] for _ in six.moves.range(n_bb)]

        # search variables
        stop_search = [False for _ in range(batch_size)]
        nbest_hyps = [[] for _ in range(batch_size)]
        ended_hyps = [[] for _ in range(batch_size)]
        vidx = None

        for i in six.moves.range(maxlen):
            vy = to_device(self, torch.LongTensor([seq[-1] for seq in yseq]))
            model_state, att_w_list, local_att_score = self.model.decode_from_state_batch(model_state, h, vy, batch_size, self.beam_size, vidx)
            local_scores = self.att_weight * local_att_score

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
                local_scores += recog_args.lm_weight * local_lm_scores

            # ctc
            if has_ctc:
                for idx in range(num_ctc):
                    att_w = att_w_list[idx]
                    att_w = att_w if isinstance(att_w, torch.Tensor) else att_w[0]
                    ctc_state[idx], local_ctc_scores = ctc_scorer[idx](yseq, ctc_state[idx], local_scores, att_w)
                    # ctc_state[idx], local_ctc_scores = ctc_scorer[idx](yseq, ctc_state[idx], local_scores, None)
                    local_scores += ctc_weight * weights_ctc_dec[idx] * local_ctc_scores

            local_scores = local_scores.view(batch_size, self.beam_size, self.odim)
            if i == 0:
                local_scores[:, 1:, :] = self.logzero

            # accumulate scores
            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch_size, self.beam_size, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch_size, -1)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, self.beam_size, 1)
            accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            accum_padded_beam_ids = (torch.div(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist()

            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_device(self, torch.LongTensor(accum_padded_beam_ids))

            # pick ended hyps
            if i >= minlen:
                penalty = (i + 1) * self.penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch_size):
                    # some inputs get stopped early
                    if stop_search[samp_i]:
                        continue
                    for beam_j in six.moves.range(self.beam_size):
                        k = samp_i * self.beam_size + beam_j
                        _vscore = None
                        if eos_vscores[samp_i][beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            if len(yk) <= min(hlen[samp_i] for hlen in hlens):
                                _vscore = eos_vscores[samp_i][beam_j] + penalty
                        elif i == maxlen - 1:
                            yk = yseq[k][:]
                            _vscore = vscores[samp_i][beam_j] + penalty
                        if _vscore:
                            yk.append(self.eos)
                            if rnnlm:
                                _vscore += recog_args.lm_weight * rnnlm.final(rnnlm_state, index=k)
                            _score = _vscore.data.cpu().numpy()
                            ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score})

            # end detection
            stop_search = [
                stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                for samp_i in six.moves.range(batch_size)
            ]
            if all(stop_search):
                break

            if rnnlm:
                rnnlm_state = self._index_select_lm_state(rnnlm_state, 0, vidx)
            if ctc_scorer[0]:
                for idx in range(self.num_encs):
                    ctc_state[idx] = ctc_scorer[idx].index_select_state(ctc_state[idx], accum_best_ids)

        torch.cuda.empty_cache()

        dummy_hyps = [{'yseq': [self.sos, self.eos], 'score': np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch_size)]
        if normalize_score:
            for samp_i in six.moves.range(batch_size):
                for x in ended_hyps[samp_i]:
                    x['score'] /= len(x['yseq'])

        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in six.moves.range(batch_size)]


        return nbest_hyps

    @staticmethod
    def _index_select_lm_state(rnnlm_state, dim, vidx):
        new_state = None
        if isinstance(rnnlm_state, dict):
            new_state = {
                k:[torch.index_select(vi, dim, vidx) for vi in v]
                for k, v in rnnlm_state.items()
            }
        elif isinstance(rnnlm_state, list):
            new_state = [rnnlm_state[int(i)][:] for i in vidx]
        assert(new_state is not None)
        return new_state

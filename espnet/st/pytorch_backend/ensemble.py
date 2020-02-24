#!/usr/bin/env python3
# encoding: utf-8

"""Ensemble decoding definition for the speech translation task."""

from espnet.nets.st_interface import STInterface
import torch
from espnet.nets.pytorch_backend.rnn.decoders import BeamSearch


class Ensemble(STInterface, torch.nn.Module):
    """Ensemble model for ST task.

    Args:
        models (STInterface): a list of modele to aggreagate.

    """

    def __init__(self, models):
        """Construct an Ensemble object."""
        super(Ensemble, self).__init__()
        torch.nn.Module.__init__(self)

        for m in models:
            self.add_module(m)
        self.models = models

    def translate(self, x, trans_args, char_list=None, rnnlm=None):
        """Construct an Ensemble object."""
        beam_search = BeamSearch([m.dec for m in self.models])
        hs = [m.encode(x) for m in self.models]
        nbest_hyps = beam_search.recognize_beam(x, trans_args, char_list, rnnlm)
        # def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None, strm_idx=0):
        # for m in self.models:
        #     nbest_hyps.extend(m.translate(x, trans_args, char_list, rnnlm))

        return nbest_hyps

    def translate_batch(self, x, trans_args, char_list=None, rnnlm=None):
        """Construct an Ensemble object."""
        nbest_hyps = []
        for m in self.models:
            nbest_hyps.append(m.translate_batch(x, trans_args, char_list, rnnlm))

        # transpose and join the results for each element of the batch
        nbest_hyps = [sum(x, []) for x in map(list, list(zip(*nbest_hyps)))]
        return nbest_hyps

#!/usr/bin/env python3
# encoding: utf-8

"""Ensemble decoding definition for the speech translation task."""

from espnet.nets.st_interface import STInterface
import torch
class Ensemble(STInterface, torch.nn.Module):
    """Ensemble model for ST task.

    Args:
        models (STInterface): a list of modele to aggreagate.

    """

    def __init__(self, models):
        """Construce an Ensemble object."""
        super(Ensemble, self).__init__()
        torch.nn.Module.__init__(self)

        for m in models:
            self.add_module(m)
        self.models = models

    def translate(self, x, trans_args, char_list=None, rnnlm=None):
        """Construce an Ensemble object."""
        nbest_hyps = []
        for m in self.models:
            nbest_hyps.append(m.translate(x, trans_args, char_list, rnnlm))

        return nbest_hyps

    def translate_batch(self, x, trans_args, char_list=None, rnnlm=None):
        """Construce an Ensemble object."""
        nbest_hyps = []
        for m in self.models:
            nbest_hyps.append(m.translate_batch(x, trans_args, char_list, rnnlm))

        # transpose
        nbest_hyps = map(list, list(zip(*nbest_hyps)))
        return nbest_hyps

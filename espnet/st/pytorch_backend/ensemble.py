class Ensemble(STInterface, torch.nn.Module):

    def __init__(self, models):
        super(Ensemble, self).__init__()
        torch.nn.Module.__init__(self)

        for m in models:
            self.add_module(m)
        self.models = models

    def translate(self, x, trans_args, char_list=None, rnnlm=None):
        nbest_hyps = []
        for m in self.models:
            nbest_hyps.append(m.translate(x, trans_args, char_list, rnnlm))

        return nbest_hyps

    def translate_batch(self, x, trans_args, char_list=None, rnnlm=None):
        nbest_hyps = []
        for m in self.models:
            nbest_hyps.append(m.translate_batch(x, trans_args, char_list, rnnlm))

        # transpose
        nbest_hyps = map(list, list(zip(*nbest_hyps)))
        return nbest_hyps

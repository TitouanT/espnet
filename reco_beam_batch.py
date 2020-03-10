def recognize_beam_batch(
	self, h, hlens, lpz, recog_args, char_list,
	rnnlm=None, normalize_score=True, strm_idx=0, lang_ids=None
):
	# to support mutiple encoder asr mode, in single encoder mode, convert torch.Tensor to List of torch.Tensor

	# n_bb = batch_size * self.beam_size  # maximal number of hypothesis
	# pad_b = to_device(self, torch.arange(batch_size) * self.beam_size).view(-1, 1)

	# max_hlen = np.amin([max(hlens[idx]) for idx in range(self.num_encs)])
	# if recog_args.maxlenratio == 0:
	# 	maxlen = max_hlen
	# else:
	# 	maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
	# minlen = int(recog_args.minlenratio * max_hlen)

	# initialization
	# c_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
	# z_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
	# # c_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
	# # z_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
	# vscores = to_device(self, torch.zeros(batch_size, self.beam_size))

	# rnnlm_state = None
	# if self.num_encs == 1:
	# 	a_prev = [None]
	# 	att_w_list, ctc_scorer, ctc_state = [None], [None], [None]
	# 	self.att[att_idx].reset()  # reset pre-computation of h
	# else:
	# 	a_prev = [None] * (self.num_encs + 1)  # atts + han
	# 	att_w_list = [None] * (self.num_encs + 1)  # atts + han
	# 	att_c_list = [None] * (self.num_encs)  # atts
	# 	ctc_scorer, ctc_state = [None] * (self.num_encs), [None] * (self.num_encs)
	# 	for idx in range(self.num_encs + 1):
	# 		self.att[idx].reset()  # reset pre-computation of h in atts and han

	# if self.replace_sos and recog_args.tgt_lang:
	# 	tgt_lang_idx = char_list.index(recog_args.tgt_lang)
	# 	yseq = [[tgt_lang_idx] for _ in six.moves.range(n_bb)]
	# elif lang_ids is not None:
	# 	# NOTE: used for evaluation during training
	# 	yseq = [[lang_ids[b // recog_args.self.beam_size]] for b in six.moves.range(n_bb)]
	# else:
	# 	yseq = [[self.sos] for _ in six.moves.range(n_bb)]

	# stop_search = [False for _ in six.moves.range(batch_size)]
	# stop_search = [False] * batch_size
	# nbest_hyps = [[] for _ in six.moves.range(batch_size)]
	# ended_hyps = [[] for _ in range(batch_size)]

	# exp_hlens = [
	# 	hlen.repeat(self.beam_size).view(self.beam_size, batch_size).transpose(0, 1).reshape(-1).tolist()
	# 	for hlen in hlens
	# ]
	# exp_h = [
	# 	hi.unsqueeze(1).repeat(1, self.beam_size, 1, 1).reshape(n_bb, *hi.size()[1:3])
	# 	for hi in h
	# ]
	# # exp_hlens = [exp_hlens[idx].view(-1).tolist() for idx in range(self.num_encs)]
	# exp_h = [h[idx].unsqueeze(1).repeat(1, self.beam_size, 1, 1).contiguous() for idx in range(self.num_encs)]
	# exp_h = [exp_h[idx].view(n_bb, h[idx].size()[1], h[idx].size()[2]) for idx in range(self.num_encs)]

	# if lpz[0] is not None:
		# scoring_ratio = CTC_SCORING_RATIO if att_weight > 0.0 and not lpz[0].is_cuda else 0
		# ctc_scorer = [
		# 	CTCPrefixScoreTH(
		# 		lpz[idx], hlens[idx], 0, self.eos, self.beam_size,
		# 		scoring_ratio = CTC_SCORING_RATIO if att_weight > 0.0 and not lpz[0].is_cuda else 0,
		# 		margin=ctc_margin
		# 	)
		# 	for idx in range(num_ctc)
		# ]

	# for i in six.moves.range(maxlen):
	# 	vy = to_device(self, torch.LongTensor(self._get_last_yseq(yseq)))
		# ey = self.dropout_emb(self.embed(vy))
		# if self.num_encs == 1:
		# 	att_c, att_w = self.att[att_idx](exp_h[0], exp_hlens[0], self.dropout_dec[0](z_prev[0]), a_prev[0])
		# 	att_w_list = [att_w]
		# else:
		# 	for idx in range(self.num_encs):
		# 		att_c_list[idx], att_w_list[idx] = self.att[idx](exp_h[idx], exp_hlens[idx],
		# 														 self.dropout_dec[0](z_prev[0]), a_prev[idx])
		# 	exp_h_han = torch.stack(att_c_list, dim=1)
		# 	att_c, att_w_list[self.num_encs] = self.att[self.num_encs](exp_h_han, [self.num_encs] * n_bb,
		# 															   self.dropout_dec[0](z_prev[0]),
		# 															   a_prev[self.num_encs])
		# ey = torch.cat((ey, att_c), dim=1)

		# attention decoder
		# z_list, c_list = self.rnn_forward(ey, z_prev, c_prev)
		# if self.context_residual:
		# 	logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
		# else:
		# 	logits = self.output(self.dropout_dec[-1](z_list[-1]))
		# local_scores = att_weight * F.log_softmax(logits, dim=1)

		# # rnnlm
		# if rnnlm:
		# 	rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
		# 	local_scores += recog_args.lm_weight * local_lm_scores

		# # ctc
		# if ctc_scorer[0]:
		# 	for idx in range(self.num_encs):
		# 		att_w = att_w_list[idx]
		# 		att_w_ = att_w if isinstance(att_w, torch.Tensor) else att_w[0]
		# 		ctc_state[idx], local_ctc_scores = ctc_scorer[idx](yseq, ctc_state[idx], local_scores, att_w_)
		# 		local_scores += ctc_weight * weights_ctc_dec[idx] * local_ctc_scores

		# local_scores = local_scores.view(batch_size, self.beam_size, self.odim)
		# if i == 0:
		# 	local_scores[:, 1:, :] = self.logzero

		# # accumulate scores
		# eos_vscores = local_scores[:, :, self.eos] + vscores
		# vscores = vscores.view(batch_size, self.beam_size, 1).repeat(1, 1, self.odim)
		# vscores[:, :, self.eos] = self.logzero
		# vscores = (vscores + local_scores).view(batch_size, -1)

		# # global pruning
		# accum_best_scores, accum_best_ids = torch.topk(vscores, self.beam_size, 1)
		# accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
		# accum_padded_beam_ids = (torch.div(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist()

		# y_prev = yseq[:][:]
		# yseq = self._index_select_list(yseq, accum_padded_beam_ids)
		# yseq = self._append_ids(yseq, accum_odim_ids)
		# vscores = accum_best_scores
		# vidx = to_device(self, torch.LongTensor(accum_padded_beam_ids))

		# a_prev = []
		# num_atts = 1 if self.num_encs == 1 else self.num_encs + 1
		# for idx in range(num_atts):
		# 	if isinstance(att_w_list[idx], torch.Tensor):
		# 		_a_prev = torch.index_select(att_w_list[idx].view(n_bb, *att_w_list[idx].shape[1:]), 0, vidx)
		# 	elif isinstance(att_w_list[idx], list):
		# 		# handle the case of multi-head attention
		# 		_a_prev = [torch.index_select(att_w_one.view(n_bb, -1), 0, vidx) for att_w_one in att_w_list[idx]]
		# 	else:
		# 		# handle the case of location_recurrent when return is a tuple
		# 		_a_prev_ = torch.index_select(att_w_list[idx][0].view(n_bb, -1), 0, vidx)
		# 		_h_prev_ = torch.index_select(att_w_list[idx][1][0].view(n_bb, -1), 0, vidx)
		# 		_c_prev_ = torch.index_select(att_w_list[idx][1][1].view(n_bb, -1), 0, vidx)
		# 		_a_prev = (_a_prev_, (_h_prev_, _c_prev_))
		# 	a_prev.append(_a_prev)
		# z_prev = [torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]
		# if self.dtype == 'lstm':
		# 	c_prev = [torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]

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
					if eos_vscores[samp_i, beam_j] > thr[samp_i]:
						yk = y_prev[k][:]
						if len(yk) <= min(hlens[idx][samp_i] for idx in range(self.num_encs)):
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
		stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
					   for samp_i in six.moves.range(batch_size)]
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

"""ScorerInterface implementation for Ensemble."""

import numpy as np
import torch

from typing import Any
from typing import List
from typing import Tuple

from espnet.nets.scorer_interface import ScorerInterface


class Model:
	def encode(self, x: torch.Tensor) -> torch.Tensor:
		return None
	def scorers(self) -> List[ScorerInterface]:
		return None

class Ensemble(torch.nn.Module):
	def __init__(self, models: List[Model]):


# def score_by_list_of_scorers(scorers, y, state, x):

class EnsembleScorer(ScorerInterface):

	def __init__(
			self,
			scorers_by_model: List[List[ScorerInterface]],
			weights_by_model: List[List[float]],
			combination_function: Callable([List[torch.Tensor]], torch.Tensor),
			part_ids_function: Callable([List[torch.Tensor]], torch.Tensor)
	):
		self.scorers_by_model = scorers_by_model
		self.weights_by_model = weights_by_model
		self.combination_function = combination_function
		self.part_ids_function = part_ids_function

	def init_state(self, x: List[torch.Tensor]):
		return [
			[scorer.init_state(xi) for scorer in scorers]
			for xi, scorers in zip(x, self.scorers_by_model)
		]

    def score(self, y: torch.Tensor, state: Any, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
		outputs = []
		new_state = []
		for i, scorers in enumerate(self.scorers_by_model):
			scorers_state = []
			scorers_ouputs = []
			weighted_scores = torch.zeros(self.n_vocab, dtype=x.dtype, device=x.device)

			for k in self.full_scorers:
				weighted_scores += self.weights[i][k] * scores[k]
			for k in self.part_scorers:
				weighted_scores[part_ids] += self.weights[k] * part_scores[k]

			ouputs.append(scorers_output)
			new_state.append(scorers_state)

		return self.combination_function(ouputs), new_state
	
    def final_score(self, state: Any) -> float:
		return 0.0
	

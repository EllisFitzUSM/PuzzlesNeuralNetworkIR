from collections import OrderedDict
from huggingface_hub import HfApi
import pyarrow
from datasets import load_dataset
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator # I think I want the reranker

import config
from _sbert_ir import BaseSBertRetriever
from ranx import Run
from torch.utils.data import DataLoader
import math
import random
import os

class SBertCE(BaseSBertRetriever):

	def __init__(self, model_name_or_path, device):
		self.device = device
		try:
			hf_api = HfApi()
			model_info = hf_api.model_info(model_name_or_path)
			self.model = CrossEncoder(model_name_or_path, device=self.device)
		except Exception as e:
			print('Model name does not exist, attempting as checkpoint.')
			if os.path.exists(model_name_or_path):
				self.model = CrossEncoder(model_name_or_path, device=self.device)
			else:
				raise Exception('Model name does not exist, and neither does the local path provided')

	def retrieve_rerank(self, run: str | dict | Run, topics, answers):
		if isinstance(run, str):
			run = Run.from_file(run, kind='trec', name=run.split('.')[0]).to_dict()
		elif isinstance(run, Run):
			run = run.to_dict()
		elif isinstance(run, dict):
			run = run
		else:
			raise TypeError('run arg is not a path or Run object')

		rerank_dict = {}
		for topic_id, ranking in run.items():
			topic = topics[topic_id]
			topic_answer_pairings = [[topic, answers[answer_id]] for answer_id in ranking]
			re_rank_scores = self.model.predict(topic_answer_pairings, batch_size=32, show_progress_bar=True)

			rerank_dict[topic_id] = dict(zip(ranking.keys(), re_rank_scores))
			rerank_dict[topic_id] = OrderedDict(sorted(rerank_dict[topic_id].items(), key=lambda t: t[1], reverse=True))

		return rerank_dict

	def fine_tune(self, train_qrel, eval_qrel, test_qrel, topics, answers, epochs: int = 4, batch_size: int = 32):
		train_samples=self.format_inputexamples(train_qrel, answers, topics)
		train_dataloader=DataLoader(train_samples,batch_size=batch_size)
		warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)

		eval_samples = self.format_dataset(eval_qrel, answers, topics)
		evaluator = CERerankingEvaluator(eval_samples, name='train')

		self.model.fit(
			train_dataloader=train_dataloader,
			evaluator=evaluator,
			epochs=epochs,
			output_path=config.sbert_ce_ft_save,
			warmup_steps=warmup_steps,
			save_best_model=True,
			use_amp=True
		)
		self.model = CrossEncoder(r'.\ce_ft')

		test_samples=self.format_dataset(test_qrel, answers, topics)
		evaluator = CERerankingEvaluator(test_samples, name='test')
		evaluator(self.model)

	def format_dataset(self, qrel, answers, topics):
		qrel_dict = qrel.to_dict()
		samples = []

		# Add our retrieved results
		for topic_id, answer_scores in qrel_dict.items():
			positive_set = set()
			negative_set = set()
			for answer_id, score in answer_scores.items():
				if score >= 1:
					positive_set.add(answers[answer_id])
				else:
					negative_set.add(answers[answer_id])

			# Add some random negatives
			for i in range(len(answer_scores)):
				random_answer_id = random.choice(list(answers.keys()))
				if random_answer_id in qrel_dict[topic_id].keys():
					continue
				negative_set.add(answers[random_answer_id])

			samples.append({'query': topics[topic_id], 'positive': list(positive_set), 'negative': list(negative_set)})
		return samples

	def format_inputexamples(self, qrel, answers, topics):
		qrel_dict = qrel.to_dict()
		input_examples = []

		# Add our retrieved results
		for topic_id, answer_scores in qrel_dict.items():
			for answer_id, score in answer_scores.items():
				if score >= 1:
					input_examples.append(InputExample(texts=[topics[topic_id], answers[answer_id]], label=1))
				else:
					input_examples.append(InputExample(texts=[topics[topic_id], answers[answer_id]], label=0))

			# Add some random negatives
			for i in range(len(answer_scores)):
				random_answer_id = random.choice(list(answers.keys()))
				if random_answer_id in qrel_dict[topic_id].keys():
					continue
				input_examples.append(InputExample(texts=[topics[topic_id], answers[random_answer_id]], label=0))

		return input_examples
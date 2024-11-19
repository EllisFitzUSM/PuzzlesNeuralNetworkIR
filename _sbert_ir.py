from sentence_transformers import SentenceTransformer, SimilarityFunction
from huggingface_hub import HfApi
import os
import config

class BaseSBertRetriever(object):

	def __init__(self, model_name_or_path, device, similarity_fn: str = 'cos'):
		self.device = device

		try:
			hf_api = HfApi()
			model_info = hf_api.model_info(model_name_or_path)
			self.model = SentenceTransformer(model_name_or_path, device=self.device, prompts=config.prompt)
		except Exception as e:
			print('Model name does not exist, attempting as checkpoint.')
			if os.path.exists(model_name_or_path):
				self.model = SentenceTransformer(model_name_or_path, device=self.device, prompts={'retrieval': 'Retrieve semantically similar text: '})
			else:
				raise Exception('Model name does not exist, and neither does the local path provided')

		try:
			self.model.prompt = config.prompt
			self.model.default_prompt_name = config.prompt_name
		except:
			pass
		match similarity_fn:
			case 'cos':
				self.model.similarity_fn_name = SimilarityFunction.COSINE
			case 'dot':
				self.model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT

	def fine_tune(self):
		raise NotImplementedError('must implement fine_tune')

	def format_dataset(self, qrel, answers, topics):
		raise NotImplementedError('must implement format_dataset')
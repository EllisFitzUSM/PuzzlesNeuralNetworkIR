from nltk.corpus import stopwords
import nltk
import os

try:
    stopwords = stopwords.words('english')
except:
    nltk.download('stopwords')
    stopwords = stopwords.words('english')

sbert_bi_model = 'sentence-transformers/msmarco-distilbert-base-v4'
sbert_bi_ft_save = r'.\bi_ft'
sbert_ce_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
sbert_ce_ft_save = r'.\ce_ft'

results_path = r'.\results'
os.makedirs(results_path, exist_ok=True)

prompt = {'retrieval': 'Retrieve semantically similar text: '}
prompt_name = 'retrieval'
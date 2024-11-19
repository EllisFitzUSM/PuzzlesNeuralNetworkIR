from xmlrpc.client import Error
from bs4 import BeautifulSoup as bs
from huggingface_hub import HfApi
from sbert_bi_ir import SBertBI
from sbert_ce_ir import SBertCE
from itertools import islice
from ranx import Run, Qrels
from tqdm import tqdm
import argparse
import random
import config
import string
import torch
import json
import re
import os

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('answers', help='Path to Answers.json')
    argparser.add_argument('topics', help='Path to topic files.', nargs='+')
    argparser.add_argument('-q', '--qrel', help='Path to qrel_1.tsv')
    argparser.add_argument('-bc', '--bi_checkpoint', help='Path to BI Checkpoint.')
    argparser.add_argument('-cc', '--ce_checkpoint', help='Path to CE Checkpoint.')
    argparser.add_argument('-qs', '--qrel_splits', help='Qrel splits.', nargs=3)
    args = argparser.parse_args()

    answers = read_answers(args.answers)
    topics_1 = read_topics(args.topics[0])
    print('Splitting Train, Evaluation, and Test...')
    train_qrel, eval_qrel, test_qrel = split_qrel(args.qrel, args.qrel_splits)
    print('Split Train, Evaluation, and Test...')
    test_topics = {topic_id: topics_1[topic_id] for topic_id in test_qrel.to_dict()}
    print('Formatted Test Topics.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Retrieving results for pretrained models on the Test Topics')
    sbert_bi, sbert_ce = get_pretrained(device)
    bi_test_run, ce_test_run = rank_rerank(sbert_bi, sbert_ce, test_topics, answers)
    save_runs(bi_test_run, ce_test_run, 'test')

    print('Retrieving results for fine-funed models on the Test Topics')
    print('If not loading models from disk, they will be fine-tuned ASAP, this may take awhile!')
    sbert_ft_bi, sbert_ft_ce = load_or_fine_tune(args, device, train_qrel, eval_qrel, test_qrel, topics_1, answers)
    bi_ft_test_run, ce_ft_test_run = rank_rerank(sbert_ft_bi, sbert_ft_ce, test_topics, answers)
    save_runs(bi_ft_test_run, ce_ft_test_run, 'ft_test')

    for topic_index, topic_path in enumerate(tqdm(args.topics[1:], desc='Retrieving Results for Additional Topics...')):
        ext = str(topic_index + 2)
        topics = read_topics(topic_path)
        bi_topic_run, ce_topic_run = rank_rerank(sbert_bi, sbert_ce, topics, answers)
        save_runs(bi_topic_run, ce_topic_run, ext)
        bi_ft_topic_run, ce_ft_topic_run = rank_rerank(sbert_ft_bi, sbert_ft_ce, topics, answers)
        save_runs(bi_ft_topic_run, ce_ft_topic_run, 'ft_' + ext)

def get_pretrained(device):
    return SBertBI(config.sbert_bi_model, device), SBertCE(config.sbert_ce_model, device)

def load_or_fine_tune(args, device, train_qrel, eval_qrel, test_qrel, topics, answers):
    if args.bi_checkpoint:
        sbert_ft_bi = SBertBI(args.bi_checkpoint, device)
    else:
        sbert_ft_bi = SBertBI(config.sbert_bi_model, device)
        sbert_ft_bi.fine_tune(train_qrel, eval_qrel, test_qrel, topics, answers)

    if args.ce_checkpoint:
        sbert_ft_ce = SBertCE(args.ce_checkpoint, device)
    else:
        sbert_ft_ce = SBertCE(config.sbert_ce_model, device)
        sbert_ft_ce.fine_tune(train_qrel, eval_qrel, test_qrel, topics, answers)
    return sbert_ft_bi, sbert_ft_ce

def rank_rerank(bi_model, ce_model, topics, answers, name_extension: str = ''):
    bi_rankings = bi_model.retrieve_rank(topics, answers)
    bi_run = Run.from_dict(bi_rankings, config.sbert_bi_model + name_extension)
    ce_rankings = ce_model.retrieve_rerank(bi_run, topics, answers)
    ce_run = Run.from_dict(ce_rankings, config.sbert_ce_model + name_extension)
    return bi_run, ce_run

def save_runs(bi_run, ce_run, ext):
    bi_run.save(os.path.join(config.results_path, f'result_bi_{ext}.tsv'), kind='trec')
    ce_run.save(os.path.join(config.results_path, f'result_ce_{ext}.tsv'), kind='trec')

# Given Answer/Document File Path, Return Document Dictionary [ DocID -> Text ]
def read_answers(answer_filepath):
    answer_list = json.load(open(answer_filepath, 'r', encoding='utf-8'))
    answer_dict = {}
    for answer in tqdm(answer_list, desc='Reading Answer Collection...', colour='yellow'):
        answer_dict[answer['Id']] = preprocess_text(answer['Text'])
    return answer_dict

def read_topics(topic_filepath: str):
    topic_list = json.load(open(topic_filepath, 'r', encoding='utf-8'))
    topic_dict = {}
    for topic in tqdm(topic_list, desc='Reading Topic Collection...', colour='blue'):
        topic_dict[topic['Id']] = preprocess_text(' '.join([topic['Title'], topic['Body'], topic['Tags']])) # Maybe remove this?
    return topic_dict

def split_qrel(qrel_filepath, qrel_splits, split: float = 0.9):
    if qrel_splits:
        train_qrel, eval_qrel, test_qrel = map(lambda q: Qrels.from_file(q, kind='trec'), qrel_splits)
        return train_qrel, eval_qrel, test_qrel
    elif qrel_filepath:
        qrel_dict = Qrels.from_file(qrel_filepath, kind='trec').to_dict()
        topic_ids = list(qrel_dict.keys())
        random.shuffle(topic_ids)
        qrel_dict = {query_id:qrel_dict[query_id] for query_id in topic_ids}

        query_count = len(qrel_dict)
        train_set_count = int(query_count * split)
        val_set_count = int((query_count - train_set_count) / 2)

        train_qrel = Qrels.from_dict(dict(islice(qrel_dict.items(), train_set_count)))
        eval_qrel = Qrels.from_dict(dict(islice(qrel_dict.items(), train_set_count, train_set_count + val_set_count)))
        test_qrel = Qrels.from_dict(dict(islice(qrel_dict.items(), train_set_count + val_set_count, None)))

        os.makedirs(r'.\qrel_splits', exist_ok=True)
        train_qrel.save(r'.\qrel_splits\train_qrel.tsv', kind='trec')
        eval_qrel.save(r'.\qrel_splits\eval_qrel.tsv', kind='trec')
        test_qrel.save(r'.\qrel_splits\test_qrel.tsv', kind='trec')

        return train_qrel, eval_qrel, test_qrel
    else:
        raise Exception('Must supply either a single Qrel to be split, or split Qrels (Train, Validation, Test)')

def preprocess_text(text_string):
    res_str = bs(text_string, "html.parser").get_text(separator=' ')
    res_str = re.sub(r'http(s)?://\S+', ' ', res_str)
    res_str = re.sub(r'[^\x00-\x7F]+', '', res_str)
    res_str = res_str.translate({ord(p): ' ' if p in r'\/.!?-_' else None for p in string.punctuation})
    res_str = ' '.join([word for word in res_str.split() if word not in config.stopwords])
    return res_str

if __name__ == '__main__':
	main()
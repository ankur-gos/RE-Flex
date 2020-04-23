"""
Experiment configuration for:
Model: RE-Flex
Benchmark: Google-RE
"""
import fasttext
import spacy
from reflex.reflex_runner import ReflexRunner
from reflex.utils import setup_experiment
from reflex.metrics import calculate_final_em_f1
import os

ex = setup_experiment('RE-Flex Google-RE')

@ex.config
def conf():
    model_dir = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large') # Path to trained weights
    model_name = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large/model.pt')
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/googlere_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/Google_RE') # Path to underlying data
    batch_size = 16
    must_choose_answer = True
    device = 'cpu'
    ls = [0]
    k = 16
    word_embeddings_path = os.path.join(os.environ['BASE_PATH'], 'weights/crawl-300d-2M-subword.bin')

@ex.automain
def main(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, word_embeddings_path, ls,  k):
    spacy_model = spacy.load('en_core_web_lg')
    we_model = fasttext.load_model(word_embeddings_path)
    per_relation_metricss = []
    for l in ls:
        runner = ReflexRunner(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, l, we_model, spacy_model, k)
        em, f1, per_relation_metrics = runner.predict()
        per_relation_metricss.append(per_relation_metrics)
    em, f1, per_relation_metrics = calculate_final_em_f1(per_relation_metricss)
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


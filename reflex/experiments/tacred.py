"""
Experiment configuration for:
Model: RE-Flex
Benchmark: TACRED
"""
import fasttext
import spacy
import pickle
from reflex.reflex_runner import ReflexRunner
from reflex.utils import setup_experiment, save_reflex_e_list
import os

ex = setup_experiment('RE-Flex TACRED')

@ex.config
def conf():
    model_dir = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large') # Path to trained weights
    model_name = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large/model.pt')
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/tacred_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/tacred/test') # Path to underlying data
    hyperparam_path = os.path.join(os.environ['BASE_PATH'], 'tacred_tune.pkl')
    error_path = os.path.join(os.environ['BASE_PATH'], 'figures', 'reflex_tacred.csv')
    batch_size = 16
    must_choose_answer = False
    device = 'cuda'
    k = 16
    override_expand = False
    override_expand_value = False
    word_embeddings_path = os.path.join(os.environ['BASE_PATH'], 'weights/crawl-300d-2M-subword.bin')

@ex.automain
def main(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, word_embeddings_path, k, hyperparam_path, error_path, override_expand, override_expand_value):
    with open(hyperparam_path, 'rb') as rf:
        hyperparams = pickle.load(rf)

    spacy_model = spacy.load('en_core_web_lg')
    we_model = fasttext.load_model(word_embeddings_path)
    runner = ReflexRunner(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, 0, we_model, spacy_model, k, override_expand_value, hyperparams=hyperparams)
    runner.override_expand = override_expand
    em, f1, per_relation_metrics = runner.predict()
    save_reflex_e_list(runner.e_list, error_path)
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


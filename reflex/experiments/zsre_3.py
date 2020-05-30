"""
Experiment configuration for:
Model: RE-Flex
Benchmark: zsre
"""
import fasttext
import spacy
from reflex.reflex_runner import ReflexRunner
from reflex.utils import setup_experiment, save_reflex_e_list
import pickle
import os

ex = setup_experiment('RE-Flex zsre 3 relations lambda')

@ex.config
def conf():
    model_dir = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large') # Path to trained weights
    model_name = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large/model.pt')
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/zsre_3_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/zsre/test') # Path to underlying data
    hyperparam_path = os.path.join(os.environ['BASE_PATH'], 'zsre_tune.pkl')
    error_path = os.path.join(os.environ['BASE_PATH'], 'figures', 'reflex_zsre2.csv')
    batch_size = 16
    must_choose_answer = False
    device = 'cuda'
    k = 16
    ls = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    override_expand = False
    override_expand_value = False
    word_embeddings_path = os.path.join(os.environ['BASE_PATH'], 'weights/crawl-300d-2M-subword.bin')

@ex.automain
def main(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, word_embeddings_path, k, hyperparam_path, error_path, override_expand, override_expand_value, ls):
    with open(hyperparam_path, 'rb') as rf:
        hyperparams = pickle.load(rf)

    spacy_model = spacy.load('en_core_web_lg')
    we_model = fasttext.load_model(word_embeddings_path)
    runner = ReflexRunner(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, 0, we_model, spacy_model, k, override_expand_value, hyperparams=hyperparams)
    runner.override_l = True
    result_dict = {}
    for l in ls:
        runner.update_l(l)
        em, f1, per_relation_metrics = runner.predict()
        result_dict[l] = per_relation_metrics
    with open('zsre_3.pkl', 'wb') as wf:
        pickle.dump(result_dict, wf)
    return result_dict


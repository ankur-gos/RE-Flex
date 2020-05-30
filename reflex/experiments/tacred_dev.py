"""
Experiment configuration for:
Model: RE-Flex
Benchmark: TACRED
"""
import fasttext
import spacy
from reflex.reflex_runner import ReflexRunner
from reflex.utils import setup_experiment
from reflex.metrics import calculate_final_em_f1_dev
import pickle
import os

ex = setup_experiment('RE-Flex TACRED Dev tuning')

@ex.config
def conf():
    model_dir = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large') # Path to trained weights
    model_name = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large/model.pt')
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/tacred_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/tacred/dev') # Path to underlying data
    batch_size = 16
    must_choose_answer = False
    device = 'cuda'
    ls = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    k = 16
    word_embeddings_path = os.path.join(os.environ['BASE_PATH'], 'weights/crawl-300d-2M-subword.bin')

@ex.automain
def main(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, word_embeddings_path, ls,  k):
    spacy_model = spacy.load('en_core_web_lg')
    we_model = fasttext.load_model(word_embeddings_path)
    runner = ReflexRunner(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, ls[0], we_model, spacy_model, k, False)
    new_ls = []
    per_relation_metricss = []
    for l in ls:
        for expand in [True, False]:
            runner.update_l(l)
            runner.expand = expand
            em, f1, per_relation_metrics = runner.predict()
            per_relation_metricss.append(per_relation_metrics)
            new_ls.append(l)
    em, f1, per_relation_metrics = calculate_final_em_f1_dev(per_relation_metricss, new_ls, [True, False] * len(ls))
    # Pickle the best l for each relation
    with open('tacred_tune.pkl', 'wb') as wf:
        pickle.dump(per_relation_metrics, wf)

    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


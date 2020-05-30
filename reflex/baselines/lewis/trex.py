"""
Experiment configuration for:
Model: Lewis et al 2019 -- https://arxiv.org/abs/1906.04980
Benchmark: T-REx
"""
from reflex.qa_runner import QARunner
from reflex.utils import setup_experiment, save_se_list
import os

ex = setup_experiment('Lewis T-REx')

@ex.config
def conf():
    qa_path = os.path.join(os.environ['BASE_PATH'], 'weights/lewis-latest') # Path to trained weights
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/trex_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/trex') # Path to underlying data
    error_path = os.path.join(os.environ['BASE_PATH'], 'figures', 'lewis_trex.csv')
    batch_size = 16
    must_choose_answer = True
    device = 'cuda'
    trained_to_reject = False

@ex.automain
def main(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer, device, trained_to_reject, error_path):
    runner = QARunner(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer, device, trained_to_reject)
    em, f1, per_relation_metrics = runner.predict()
    save_se_list(runner.se_list, error_path)
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


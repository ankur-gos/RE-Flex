"""
Experiment configuration for:
Model: BiDAF trained on Squad2.0
Benchmark: Tacred
"""
from reflex.bidaf_runner import BidafRunner
from reflex.utils import setup_experiment, save_se_list
import os

ex = setup_experiment('BiDAF GoogleRE')

@ex.config
def conf():
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/googlere_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/googlere') # Path to underlying data
    error_path = os.path.join(os.environ['BASE_PATH'], 'figures', 'bidaf_googlere.csv')
    must_choose_answer = True
    calculate_single_error = True

@ex.automain
def main(relations_filepath, data_directory, must_choose_answer, error_path, calculate_single_error):
    runner = BidafRunner(relations_filepath, data_directory, must_choose_answer, calculate_single_error)
    em, f1, per_relation_metrics = runner.predict()
    save_se_list(runner.se_list, error_path)
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}



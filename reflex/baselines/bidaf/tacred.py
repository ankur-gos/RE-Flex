"""
Experiment configuration for:
Model: BiDAF trained on Squad2.0
Benchmark: Tacred
"""
from reflex.bidaf_runner import BidafRunner
from reflex.utils import setup_experiment
import os

ex = setup_experiment('BiDAF TACRED')

@ex.config
def conf():
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/tacred_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/tacred/test') # Path to underlying data
    must_choose_answer = False
    calculate_single_error = False

@ex.automain
def main(relations_filepath, data_directory, must_choose_answer, calculate_single_error):
    runner = BidafRunner(relations_filepath, data_directory, must_choose_answer, calculate_single_error)
    em, f1, per_relation_metrics = runner.predict()
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}



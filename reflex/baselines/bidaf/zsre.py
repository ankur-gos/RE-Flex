"""
Experiment configuration for:
Model: BiDAF trained on Squad2.0
Benchmark: ZSRE
"""
from reflex.bidaf_runner import BidafRunner
from reflex.utils import setup_experiment
import os

ex = setup_experiment('BiDAF ZSRE')

@ex.config
def conf():
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/zsre_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/zsre/test') # Path to underlying data
    must_choose_answer = False

@ex.automain
def main(relations_filepath, data_directory, must_choose_answer):
    runner = BidafRunner(relations_filepath, data_directory, must_choose_answer)
    em, f1, per_relation_metrics = runner.predict()
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}



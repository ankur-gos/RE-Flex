"""
Experiment configuration for:
Model: Radford Roberta (Naive w/ context seed)
Benchmark: T-REx
"""
from reflex.lm_runner import LMRunner
from reflex.utils import setup_experiment
import os

ex = setup_experiment('Radford T-REx')

@ex.config
def conf():
    model_dir = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large') # Path to trained weights
    model_name = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large/model.pt')
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/trex_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/trex') # Path to underlying data
    batch_size = 16
    must_choose_answer = True
    use_context = True
    device = 'cuda'
    cap = 10

@ex.automain
def main(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, use_context, cap):
    runner = LMRunner(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, cap)
    em, f1, per_relation_metrics = runner.predict_naive(use_context=use_context)
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


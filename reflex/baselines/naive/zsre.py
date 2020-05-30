"""
Experiment configuration for:
Model: Naive RoBERTa (no context seed)
Benchmark: ZSRE
"""
from reflex.lm_runner import LMRunner
from reflex.utils import setup_experiment
import os

ex = setup_experiment('Naive ZSRE')

@ex.config
def conf():
    model_dir = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large') # Path to trained weights
    model_name = os.path.join(os.environ['BASE_PATH'], 'weights/roberta_large/model.pt')
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/zsre_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/zsre/test') # Path to underlying data
    batch_size = 16
    must_choose_answer = True
    use_context = False
    device = 'cuda'
    cap = 0

@ex.automain
def main(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, use_context, cap):
    runner = LMRunner(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, cap)
    em, f1, per_relation_metrics = runner.predict_naive(use_context=False)
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


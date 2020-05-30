"""
Experiment configuration for:
Model: Dhingra et al 2018 -- https://arxiv.org/abs/1804.00720
Benchmark: ZSRE
"""
from reflex.qa_runner import QARunner
from reflex.utils import setup_experiment
import os

ex = setup_experiment('Dhingra ZSRE')


@ex.config
def conf():
    qa_path = os.path.join(os.environ['BASE_PATH'], 'weights/dhingra-latest') # Path to trained weights
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/zsre_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/zsre/test') # Path to underlying data
    batch_size = 16
    must_choose_answer = True
    device = 'cuda'
    trained_to_reject = False

@ex.automain
def main(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer, device, trained_to_reject):
    runner = QARunner(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer, device, trained_to_reject, calculate_single_error=False)
    em, f1, per_relation_metrics = runner.predict()
    print(f'Total samples: {runner.total_samples}')
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


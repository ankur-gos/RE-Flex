"""
Experiment configuration for:
Model: BERT trained on ZSRE
Benchmark: Google-RE
"""
from reflex.qa_runner import QARunner
from reflex.utils import setup_experiment, save_se_list
import os

ex = setup_experiment('BERT ZSRE GoogleRE')

@ex.config
def conf():
    qa_path = os.path.join(os.environ['BASE_PATH'], 'weights/bert-zsre') # Path to trained weights
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/googlere_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/googlere') # Path to underlying data
    error_path = os.path.join(os.environ['BASE_PATH'], 'figures', 'bzsre_googlere.csv')
    batch_size = 16
    must_choose_answer = True
    device = 'cuda'
    trained_to_reject = True

@ex.automain
def main(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer, device, trained_to_reject, error_path):
    runner = QARunner(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer, device, trained_to_reject)
    em, f1, per_relation_metrics = runner.predict()
    save_se_list(runner.se_list, error_path)
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


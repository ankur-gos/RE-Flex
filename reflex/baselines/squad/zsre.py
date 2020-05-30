"""
Experiment configuration for:
Model: BERT trained squad
Benchmark: ZSRE
"""
from reflex.qa_runner import QARunner
from reflex.utils import setup_experiment
import os
import pickle

ex = setup_experiment('BERT Squad2.0 ZSRE')

@ex.config
def conf():
    qa_path = os.path.join(os.environ['BASE_PATH'], 'weights/squad2') # Path to trained weights
    relations_filepath = os.path.join(os.environ['BASE_PATH'], 'data/zsre_relations.jsonl') # Path to relations file
    data_directory = os.path.join(os.environ['BASE_PATH'], 'data/zsre/test') # Path to underlying data
    batch_size = 16
    must_choose_answer = False
    device = 'cuda'
    trained_to_reject = True

@ex.automain
def main(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer, device, trained_to_reject):
    runner = QARunner(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer, device, trained_to_reject, calculate_single_error=False)
    em, f1, per_relation_metrics = runner.predict()
    with open('BERTSQUADZSRE.pkl', 'wb') as wf:
        pickle.dump(per_relation_metrics, wf)
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}



"""
Experiment configuration for:
Model: BERT trained on ZSRE
Benchmark: ZSRE
"""
from reflex.qa_runner import QARunner
from reflex.utils import setup_experiment

ex = setup_experiment('BERT ZSRE ZSRE')

@ex.config
def conf():
    qa_path = '/Users/ankur/Projects/RE-Flex/weights/zsre' # Path to trained weights
    relations_filepath = '/Users/ankur/Projects/RE-Flex/data/zsre_relations.jsonl' # Path to relations file
    data_directory = '/Users/ankur/Projects/RE-Flex/data/zsre' # Path to underlying data
    batch_size = 16
    must_choose_answer = True

@ex.automain
def main(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer):
    runner = QARunner(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer)
    em, f1, per_relation_metrics = runner.predict()
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


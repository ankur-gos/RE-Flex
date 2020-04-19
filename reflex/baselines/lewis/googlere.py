"""
Experiment configuration for:
Model: Lewis et al 2019 -- https://arxiv.org/abs/1906.04980
Benchmark: Google-RE
"""
from reflex.qa_runner import QARunner
from reflex.utils import setup_experiment

ex = setup_experiment('Lewis Google-RE')

@ex.config
def conf():
    qa_path = '/Users/ankur/Projects/RE-Flex/weights/lewis-latest' # Path to trained weights
    relations_filepath = '/Users/ankur/Projects/RE-Flex/data/googlere_relations.jsonl' # Path to relations file
    data_directory = '/Users/ankur/Projects/RE-Flex/data/Google_RE2' # Path to underlying data
    batch_size = 16
    must_choose_answer = True

@ex.automain
def main(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer):
    runner = QARunner(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer)
    em, f1, per_relation_metrics = runner.predict()
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


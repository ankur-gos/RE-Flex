"""
Experiment configuration for:
Model: BERT trained squad
Benchmark: ZSRE
"""
from sacred import Experiment
from sacred.observers import MongoObserver, SlackObserver
from reflex.qa_runner import QARunner
mongo_uri = 'mongodb://mongo_user:mongo_password@localhost:27017/sacred?authSource=admin'
ex = Experiment('BERT Squad2.0 ZSRE')
ex.observers.append(MongoObserver(url=mongo_uri,
                                      db_name='sacred'))
slack_obs = SlackObserver.from_config('/Users/ankur/configs/slack.json')
ex.observers.append(slack_obs)

@ex.config
def conf():
    qa_path = '/Users/ankur/Projects/RE-Flex/weights/squad2' # Path to trained weights
    relations_filepath = '/Users/ankur/Projects/RE-Flex/data/zsre_relations.jsonl' # Path to relations file
    data_directory = '/Users/ankur/Projects/RE-Flex/data/zsre' # Path to underlying data
    batch_size = 16
    must_choose_answer = True

@ex.automain
def main(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer):
    runner = QARunner(qa_path, relations_filepath, data_directory, batch_size, must_choose_answer)
    em, f1, per_relation_metrics = runner.predict()
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


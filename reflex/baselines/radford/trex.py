"""
Experiment configuration for:
Model: Radford Roberta (Naive w/ context seed)
Benchmark: T-REx
"""
from sacred import Experiment
from sacred.observers import MongoObserver, SlackObserver
from reflex.lm_runner import LMRunner
mongo_uri = 'mongodb://mongo_user:mongo_password@localhost:27017/sacred?authSource=admin'
ex = Experiment('Radford T-REx')
ex.observers.append(MongoObserver(url=mongo_uri,
                                      db_name='sacred'))
slack_obs = SlackObserver.from_config('/Users/ankur/configs/slack.json')
ex.observers.append(slack_obs)

@ex.config
def conf():
    model_dir = '/Users/ankur/Projects/RE-Flex/weights/roberta_large' # Path to trained weights
    model_name = '/Users/ankur/Projects/RE-Flex/weights/roberta_large/model.pt'
    relations_filepath = '/Users/ankur/Projects/RE-Flex/data/trex_relations.jsonl' # Path to relations file
    data_directory = '/Users/ankur/Projects/RE-Flex/data/trex' # Path to underlying data
    batch_size = 16
    must_choose_answer = True
    use_context = True
    device = 'cpu'

@ex.automain
def main(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, use_context):
    runner = LMRunner(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer)
    em, f1, per_relation_metrics = runner.predict_naive(use_context=False)
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


"""
Experiment configuration for:
Model: RE-Flex
Benchmark: Google-RE
"""
from sacred import Experiment
from sacred.observers import MongoObserver, SlackObserver
from reflex.reflex_runner import ReflexRunner
mongo_uri = 'mongodb://mongo_user:mongo_password@localhost:27017/sacred?authSource=admin'
ex = Experiment('RE-Flex Test')
ex.observers.append(MongoObserver(url=mongo_uri,
                                      db_name='sacred'))
slack_obs = SlackObserver.from_config('/Users/ankur/configs/slack.json')
ex.observers.append(slack_obs)

@ex.config
def conf():
    model_dir = '/Users/ankur/Projects/RE-Flex/weights/roberta_large' # Path to trained weights
    model_name = '/Users/ankur/Projects/RE-Flex/weights/roberta_large/model.pt'
    relations_filepath = '/Users/ankur/Projects/RE-Flex/data/test_relations.jsonl' # Path to relations file
    data_directory = '/Users/ankur/Projects/RE-Flex/data/Test' # Path to underlying data
    batch_size = 16
    must_choose_answer = True
    device = 'cpu'
    l = -3
    k = 2
    word_embeddings_path = '/Users/ankur/Projects/RE-Flex/weights/crawl-300d-2M-subword.bin'

@ex.automain
def main(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, l, word_embeddings_path, k):
    runner = ReflexRunner(model_dir, model_name, device, relations_filepath, data_directory, batch_size, must_choose_answer, l, word_embeddings_path, k)
    em, f1, per_relation_metrics = runner.predict()
    return {'em': em, 'f1': f1, 'per_relation_metrics': per_relation_metrics}


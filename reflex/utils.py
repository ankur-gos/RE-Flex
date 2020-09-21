import json
from dataclasses import dataclass
from sacred import Experiment
from sacred.observers import MongoObserver, SlackObserver
import os
import pandas as pd
from reflex.structs import Sample

def save_se_list(se_list, filename):
    s = pd.Series(pd.Categorical(se_list, categories=['no_overlap', 'lb_1', 'lb_2', 'lb_3', 'lb_4']))
    result = s.value_counts(normalize=True)
    result.to_csv(filename)

def save_reflex_e_list(e_list, filename):
    s = pd.Series(pd.Categorical(e_list, categories=['should_reject', 'should_accept', 'no_overlap', 'span_mismatch']))
    result = s.value_counts(normalize=True)
    result.to_csv(filename)

def get_bpe_val(ind, source_dictionary, bpe_obj):
    bpe = source_dictionary[ind]
    try:
        _ = int(bpe)
    except ValueError:
        return bpe
    val = bpe_obj.decode(bpe)
    return val


def setup_experiment(experiment_name):
    mongo_uri = 'mongodb://mongo_user:mongo_password@localhost:27017/sacred?authSource=admin'
    ex = Experiment(experiment_name, save_git_info=False)
    ex.observers.append(MongoObserver(url=mongo_uri,
                                          db_name='sacred'))
    slack_obs = SlackObserver.from_config(os.environ['SLACK_CONFIG'])
    ex.observers.append(slack_obs)
    return ex


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data


def load_samples_from_file(filename, template):
    data = load_file(filename)
    samples = []
    for d in data:
        if 'object' in d:
            obj = d['object']
        else:
            obj = None
        sample = Sample(d['subject'], d['context'], obj, None, template)
        samples.append(sample)
    return samples


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


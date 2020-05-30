import os
import shutil
import subprocess
import json
from reflex.utils import load_file
import pickle

class BidafRunner:
    def __init__(self, relations_filepath, data_directory, must_choose_answer, calculate_single_error):
        self.relations_filepath = relations_filepath # path to relations file
        self.data_directory = data_directory # data directory path
        self.must_choose_answer = must_choose_answer # For datasets where there is always an answer, setting this to true will ensure that QA models that can return "answer doesn't exist" will always return a span in the context
        if calculate_single_error:
            self.se_list = []
        else:
            self.se_list = None

    def predict(self):
        # Load relations file
        relations = load_file(self.relations_filepath)
        # Iterate through relations file and predict for each relation
        aggregate_em = aggregate_f1 = 0
        per_relation_metrics = {}
        for relation in relations:
            data_file = os.path.join(self.data_directory, relation['relation']) + '_qa.json'
            shutil.copyfile(data_file, os.path.join(os.environ['BASE_PATH'], 'dataqa', 'test.json'))
            if self.must_choose_answer:
                subprocess.run(f'docker-compose -f {os.path.join(os.environ["BASE_PATH"], "reflex", "docker-compose-bidaf.yml")} run bidaf', shell=True)
            else:
                subprocess.run(f'docker-compose -f {os.path.join(os.environ["BASE_PATH"], "reflex", "docker-compose-bidaf.yml")} run bidaf_na', shell=True)

            eval_path = os.path.join(os.environ['DATA_DIR'], 'eval.json') 
            with open(eval_path) as rf:
                for l in rf.readlines():
                    l = l.strip()
                    j = json.loads(l)
                    em = j['exact'] / 100
                    f1 = j['f1'] / 100
                    per_relation_metrics[relation['relation']] = {'em': em, 'f1': f1}
                    aggregate_em += em
                    aggregate_f1 += f1
                    break
            os.remove(eval_path)
            if self.se_list is not None:
                rez_pth = os.path.join(os.environ['DATA_DIR'], 'rez.pkl')
                with open(rez_pth, 'rb') as rf:
                    self.se_list.extend(pickle.load(rf))
                os.remove(rez_pth)
        aggregate_em /= len(relations)
        aggregate_f1 /= len(relations)

        return aggregate_em, aggregate_f1, per_relation_metrics

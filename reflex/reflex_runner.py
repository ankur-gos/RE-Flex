"""
Classes for running lm inference
"""
import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from reflex.models.reflex import Reflex
from dataclasses import dataclass
from reflex.utils import load_file, to_list
from reflex.structs import Sample
from reflex.models.pmi_filter import WordEmbeddingsPMIFilter
from reflex.squad_utils import convert_examples_to_features, read_input_examples, RawResult, get_predictions
from reflex.metrics import calculate_relation_metrics
from tqdm import tqdm


class ReflexRunner:
    def __init__(self, 
                 model_dir, 
                 model_name, 
                 device, 
                 relations_filepath, 
                 data_directory, 
                 batch_size, 
                 must_choose_answer, 
                 l, 
                 word_embeddings_bin_path,
                 k):
        self.context_filter = WordEmbeddingsPMIFilter(word_embeddings_bin_path, l)
        self.model = Reflex(model_dir, model_name, device, k, self.context_filter.nlp)
        self.relations_filepath = relations_filepath # path to relations file
        self.data_directory = data_directory # data directory path
        self.batch_size = batch_size
        self.must_choose_answer = must_choose_answer # For datasets where there is always an answer, setting this to true will ensure that QA models that can return "answer doesn't exist" will always return a span in the context

    def predict(self):
        # Load relations file
        relations = load_file(self.relations_filepath)
        # Iterate through relations file and predict for each relation
        aggregate_em = aggregate_f1 = 0
        per_relation_metrics = {}
        for relation in relations:
            data_file = os.path.join(self.data_directory, relation['relation']) + '.jsonl'
            data = load_file(data_file)
            # Adding to set filters any accidental duplicates
            samples = set()
            for d in data:
                samples.add(Sample(d['subject'], d['context'], d['object'], None, relation['template']))

            samples = list(samples)
            init_len = len(samples)
            print('Starting filtering')
            samples = self.context_filter.filter(samples)
            final_len = len(samples)
            print(f'Filtering finished. Filtered {init_len - final_len}.')
            
            print(f'Loaded relation {relation["relation"]}. There are {len(samples)} test samples')
            print('Batching samples')
            batches = self.model.batch(samples, self.batch_size)
            print('Starting inference')
            all_results = []
            for batch in tqdm(batches):
                results = self.model.predict(batch)
                all_results.extend(results)
            relation_em, relation_f1, per_relation_metrics = calculate_relation_metrics(samples, all_results, per_relation_metrics, relation)
            aggregate_em += relation_em
            aggregate_f1 += relation_f1
        aggregate_em /= len(relations)
        aggregate_f1 /= len(relations)
        return aggregate_em, aggregate_f1, per_relation_metrics


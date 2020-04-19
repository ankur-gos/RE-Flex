"""
Classes for running QA inference
"""
import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
from dataclasses import dataclass
from reflex.utils import load_file
from reflex.squad_utils import convert_examples_to_features, read_input_examples, RawResult, get_predictions
from reflex.metrics import calculate_em_f1, calculate_relation_metrics
from tqdm import tqdm

@dataclass(frozen=True)
class Sample:
    head: str
    context: str
    tail: str
    question: str

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class QARunner:
    def __init__(self, qa_path, relations_filepath, data_directory, batch_size, must_choose_answer, device, trained_to_reject):
        self.trained_to_reject = trained_to_reject
        self.qa_path = qa_path # path to qa weights
        self.relations_filepath = relations_filepath # path to relations file
        self.data_directory = data_directory # data directory path
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased') # tokenizer
        self.model = BertForQuestionAnswering.from_pretrained(qa_path) # Load the model
        self.model.to(device)
        self.device = device

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
                question = relation['question'].replace('[X]', d['subject'])
                samples.add(Sample(d['subject'], d['context'], d['object'], question))
            samples = list(samples)
            print(f'Loaded relation {relation["relation"]}. There are {len(samples)} test samples')
            # Most of below is taken directly from HuggingFace, which is what Lewis et al use to train their QA head
            # Defaults from huggingface
            do_lower_case = True
            max_answer_length = 30
            verbose_logging = False
            null_score_diff_threshold = 0.0
            n_best = 20
            max_query_length = 64
            doc_stride = 128
            max_seq_length = 384

            # Load the samples into squad format
            examples = read_input_examples(samples)
            features = convert_examples_to_features(examples=examples,
                                                    tokenizer=self.tokenizer,
                                                    max_seq_length=max_seq_length,
                                                    doc_stride=doc_stride,
                                                    max_query_length=max_query_length,
                                                    is_training=False,
                                                    cls_token_segment_id=0,
                                                    pad_token_segment_id=0,
                                                    cls_token_at_end=False,
                                                    sequence_a_is_doc=False)

            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
            eval_sampler = SequentialSampler(dataset)
            eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.batch_size)
            all_results = []
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                #stime = time.time()
                batch = tuple(t.to(device=self.device) for t in batch)
                with torch.no_grad():
                    inputs = {'input_ids':      batch[0],
                              'attention_mask': batch[1]
                              }
                    inputs['token_type_ids'] = batch[2]
                    example_indices = batch[3]
                    outputs = self.model(**inputs)

                for i, example_index in enumerate(example_indices):
                    eval_feature = features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    result = RawResult(unique_id    = unique_id,
                                       start_logits = to_list(outputs[0][i]),
                                       end_logits   = to_list(outputs[1][i]))
                    all_results.append(result)

            predictions = get_predictions(examples, features, all_results, n_best,
                max_answer_length, do_lower_case,
                verbose_logging,
                self.trained_to_reject, null_score_diff_threshold, must_choose_answer=self.must_choose_answer)
            predictions = [predictions[p] for p in predictions]
            relation_em, relation_f1, per_relation_metrics = calculate_relation_metrics(samples, predictions, per_relation_metrics, relation)
            aggregate_em += relation_em
            aggregate_f1 += relation_f1
        aggregate_em /= len(relations)
        aggregate_f1 /= len(relations)
        return aggregate_em, aggregate_f1, per_relation_metrics


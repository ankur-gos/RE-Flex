# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Hello hadar
#
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                                      TensorDataset)
from retrieve_docs import retrieve_docs
from lama.modules import build_model_by_name
import pickle
import lama.utils as utils
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
from tqdm import tqdm
from random import shuffle
import os
import json
import spacy
import lama.modules.base_connector as base
from pprint import pprint
import logging.config
import logging
import pickle
from multiprocessing.pool import ThreadPool
import multiprocessing
import evaluation_metrics as metrics
import time, sys
import torch
import gc
from squad_utils import convert_examples_to_features, read_input_examples, RawResult, get_predictions
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
import run
def to_list(tensor):
    return tensor.detach().cpu().tolist()



def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def create_logdir_with_timestamp(base_logdir, modelname):
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, modelname, timestr)
    os.makedirs(log_directory)

    path = "{}/last".format(base_logdir)
    try:
        os.unlink(path)
    except Exception:
        pass
    os.symlink(log_directory, path)
    return log_directory


def parse_template(template, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]


def init_logging(log_directory):
    logger = logging.getLogger("LAMA")
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_directory, exist_ok=True)

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # file handler
    fh = logging.FileHandler(str(log_directory) + "/info.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False

    return logger


def batchify(data, batch_size):
    msg = ""
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    c = 0

    # sort to group togheter sentences with similar length
    for sample in sorted(
        data, key=lambda k: len(" ".join(k["masked_sentences"]).split())
    ):
        masked_sentences = sample["masked_sentences"]
        current_samples_batch.append(sample)
        current_sentences_batches.append(masked_sentences)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            c = 0

    # last batch
    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)

    return list_samples_batches, list_sentences_batches, msg


def run_thread(arguments):

    msg = ""

    # 1. compute the ranking metrics on the filtered log_probs tensor
    sample_MRR, sample_P, experiment_result, return_msg = metrics.get_ranking(
        arguments["mymodel_probs"],
        arguments["masked_indices"],
        arguments["vocab"],
        is_masked_probs=False,
        label_index=arguments["label_index"],
        print_generation=arguments["interactive"],
        topk=10000,
    )
    em, f1, is_error, no_overlap, larger_by_1, larger_by_2, larger_by_3, larger_by_4, larger_by_5_or_more = metrics.calculate_em_f1(arguments['target'], arguments['prediction'])
    msg += "\n" + return_msg

    sample_perplexity = 0.0
    if arguments["interactive"]:
        pprint(arguments["sample"])
        # THIS IS OPTIONAL - mainly used for debuggind reason
        # 2. compute perplexity and print predictions for the complete log_probs tensor
        sample_perplexity, return_msg = print_sentence_predictions(
            arguments["original_log_probs"],
            arguments["token_ids"],
            arguments["vocab"],
            masked_indices=arguments["masked_indices"],
            print_generation=arguments["interactive"],
        )
        input("press enter to continue...")
        msg += "\n" + return_msg

    return experiment_result, sample_MRR, sample_P, sample_perplexity, msg, em, f1, is_error, no_overlap, larger_by_1, larger_by_2, larger_by_3, larger_by_4, larger_by_5_or_more


def lowercase_samples(samples):
    new_samples = []
    for sample in samples:
        sample["obj_label"] = sample["obj_label"].lower()
        sample["sub_label"] = sample["sub_label"].lower()
        lower_masked_sentences = []
        for sentence in sample["masked_sentences"]:
            sentence = sentence.lower()
            sentence = sentence.replace(base.MASK.lower(), base.MASK)
            lower_masked_sentences.append(sentence)
        sample["masked_sentences"] = lower_masked_sentences
        new_samples.append(sample)
    return new_samples


def filter_samples(model, samples, vocab_subset, max_sentence_length, template,
            condition_on_answer_exists=False,
            condition_on_single_token=False,
            condition_on_multi_token=False,
            condition_on_answer_does_not_exist=False,
            is_zsre=False):
    msg = ""
    new_samples = []
    samples_exluded = 0
    for sample in samples:
        excluded = False
        if "obj_label" in sample and "sub_label" in sample:

            if not is_zsre:
                obj_label_ids = model.get_id(sample["obj_label"])

                if obj_label_ids is not None:
                    final_str = model.vocab[obj_label_ids[0]]
                    if len(obj_label_ids) > 1:
                        for x in obj_label_ids[1:]:
                            final_str = f'{final_str}{model.vocab[x][1:-1]}'
                    reconstructed_word = " ".join(
                        [model.vocab[x] for x in obj_label_ids]
                    ).strip()
                else:
                    reconstructed_word = None
            else:
                if not condition_on_answer_exists and not condition_on_single_token and not condition_on_multi_token and not condition_on_answer_does_not_exist:
                    reconstructed_word = sample['obj_label']
                else:
                    decomposed_obj = sample['obj_label'].split(' ')
                    decomposed_obj = [o for o in decomposed_obj if o != '']
                    if condition_on_answer_exists:
                        if len(decomposed_obj) == 0:
                            msg += "\tEXCLUDED answer must exist\n"
                            samples_exluded += 1
                            continue
                    if condition_on_answer_does_not_exist:
                        if len(decomposed_obj) > 0:
                            msg += "\tEXCLUDED answer must not exist\n"
                            samples_exluded += 1
                            continue
                    if condition_on_single_token:
                        if len(decomposed_obj) > 1:
                            msg += "\tEXCLUDED answer must be single token\n"
                            samples_exluded += 1
                            continue
                    if condition_on_multi_token:
                        if len(decomposed_obj) == 1:
                            msg += "\tEXCLUDED answer must be multi token\n"
                            samples_exluded += 1
                            continue


            
            excluded = False
            if not template or len(template) == 0:
                masked_sentences = sample["masked_sentences"]
            elif "judgments" in sample:
                # only for Google-RE
                num_no = 0
                num_yes = 0
                for x in sample["judgments"]:
                    if x["judgment"] == "yes":
                        num_yes += 1
                    else:
                        num_no += 1
                if num_no > num_yes:
                    # SKIP NEGATIVE EVIDENCE
                    pass
                else:
                    if reconstructed_word != sample["obj_label"]:
                        sample['obj_label'] = model.vocab[obj_label_ids[0]]
                        sample['reconstructed_word'] = final_str
                        new_samples.append(sample)
                    else:
                        sample['reconstructed_word'] = sample['obj_label']
                        new_samples.append(sample)
            elif reconstructed_word != sample["obj_label"]:
                if is_zsre:
                    sample['reconstructed_word'] = sample['obj_label']
                else:
                    sample['reconstructed_word'] = final_str
                sample['obj_label'] = model.vocab[obj_label_ids[0]]
                new_samples.append(sample)
            elif reconstructed_word == sample['obj_label']:
                sample['reconstructed_word'] = sample['obj_label']
                new_samples.append(sample)
            elif obj_label_ids is None:
                msg += "\tEXCLUDED object label {} not in model vocabulary\n".format(
                    sample["obj_label"]
                )
                samples_exluded += 1
            else:
                raise Exception('wat')

            # Filter
        else:
            msg += "\tEXCLUDED since 'obj_label' not sample or 'sub_label' not in sample: {}\n".format(
                sample
            )
            samples_exluded += 1
    msg += "samples exluded  : {}\n".format(samples_exluded)
    return new_samples, msg


def main(args, shuffle_data=True, model=None, qamodel=None, tokenizer=None, zsre=False, v2=True, must_choose_answer=False,
            condition_on_answer_exists=False,
            condition_on_single_token=False,
            condition_on_multi_token=False,
            condition_on_answer_does_not_exist=False):

    if len(args.models_names) > 1:
        raise ValueError('Please specify a single language model (e.g., --lm "bert").')

    msg = ""

    [model_type_name] = args.models_names

    print(model)
    #if model is None:
    #    #model = build_model_by_name(model_type_name, args)

    if model_type_name == "fairseq":
        model_name = "fairseq_{}".format(args.fairseq_model_name)
    elif model_type_name == "bert":
        model_name = "BERT_{}".format(args.bert_model_name)
    elif model_type_name == "elmo":
        model_name = "ELMo_{}".format(args.elmo_model_name)
    else:
        model_name = model_type_name.title()

    # initialize logging
    if args.full_logdir:
        log_directory = args.full_logdir
    else:
        log_directory = create_logdir_with_timestamp(args.logdir, model_name)
    logger = init_logging(log_directory)
    msg += "model name: {}\n".format(model_name)

    # deal with vocab subset
    vocab_subset = None
    index_list = None
    msg += "args: {}\n".format(args)

    logger.info("\n" + msg + "\n")

    # dump arguments on file for log
    with open("{}/args.json".format(log_directory), "w") as outfile:
        json.dump(vars(args), outfile)

    # stats
    samples_with_negative_judgement = 0
    samples_with_positive_judgement = 0

    # Mean reciprocal rank
    MRR = 0.0
    MRR_negative = 0.0
    MRR_positive = 0.0

    # Precision at (default 10)
    Precision = 0.0
    Precision1 = 0.0
    Precision_negative = 0.0
    Precision_positivie = 0.0

    # EM
    EM = 0.0

    # F1
    F1 = 0.0
    is_error = 0
    no_overlap = 0
    larger_by_1 = 0
    larger_by_2 = 0
    larger_by_3 = 0
    larger_by_4 = 0
    larger_by_5_or_more = 0
    data = load_file(args.dataset_filename)

    print(len(data))

    if args.lowercase:
        # lowercase all samples
        logger.info("lowercasing all samples...")
        all_samples = lowercase_samples(data)
    else:
        # keep samples as they are
        all_samples = data

    all_samples, ret_msg = filter_samples(
        model, data, vocab_subset, args.max_sentence_length, args.template, 
            condition_on_answer_exists=condition_on_answer_exists,
            condition_on_single_token=condition_on_single_token,
            condition_on_multi_token=condition_on_multi_token,
            condition_on_answer_does_not_exist=condition_on_answer_does_not_exist,
            is_zsre=zsre
    )

    # OUT_FILENAME = "{}.jsonl".format(args.dataset_filename)
    # with open(OUT_FILENAME, 'w') as outfile:
    #     for entry in all_samples:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')

    logger.info("\n" + ret_msg + "\n")

    print(len(all_samples))
    if len(all_samples) == 0:# or len(all_samples) >= 50:
        return None, None, None, None, None, None, None, None, None, None, None, None

    # if template is active (1) use a single example for (sub,obj) and (2) ...
    if args.template and args.template != "":
        facts = []
        sub_objs = []
        for sample in all_samples:
            sub = sample["sub_label"]
            obj = sample["obj_label"]
            target = sample['reconstructed_word']
            question = args.question
            if 'reconstructed_word' not in sample:
                raise Exception('Reconstructed word not in sample... fix this')
            else:
                if 'masked_sentences' in sample:
                    # Some of the masked sentences don't have a mask in them, need to find first with mask
                    context = None
                    for sent in sample['masked_sentences']:
                        if not zsre:
                            if '[MASK]' in sent:
                                context = sent.replace('[MASK]', sample['reconstructed_word'])
                                break
                        else:
                            context = sent
                    if context is None:
                        print('No valid context found, skipping sample')
                        continue
                else:
                    context = None
                    for evidence in sample['evidences']:
                        if not zsre:
                            if '[MASK]' in evidence['masked_sentence']:
                                context = evidence['masked_sentence'].replace('[MASK]', sample['reconstructed_word'])
                                break
                        else:
                            context = evidence['masked_sentence']
                    if context is None:
                        print('No valid context found, skipping sample')
                        continue

            #context = context.replace('(', '')
            #context = context.replace(')', '')
            if (sub, target, context) not in sub_objs:
                sub_objs.append((sub, target, context))
                if 'reconstructed_word' in sample:
                    facts.append((sub, obj, context, question, sample['reconstructed_word']))
                else:
                    facts.append((sub, obj, context, question, obj))

                #break
        local_msg = "distinct template facts: {}".format(len(facts))
        logger.info("\n" + local_msg + "\n")
        print(local_msg)
        all_samples = []
        for fact in facts:
            (sub, obj, context, question, rw) = fact
            sample = {}
            sample["sub_label"] = sub
            sample["obj_label"] = obj
            sample["reconstructed_word"] = rw
            # sobstitute all sentences with a standard template
            sample['context'] = context
            sample["masked_sentences"] = parse_template(
                args.template.strip(), sample["sub_label"].strip(), base.MASK
            )
            question = question.replace('[X]', sub)
            sample['question'] = question
            #query = sample['masked_sentences'][0].replace(base.MASK, '')
            #sample['query'] = query
            #print(f'query={query}')
            #docs = retrieve_docs(query, ranker, conn, 30)
            #sample['context'] = docs[0]
            #print(f'docs={docs}')
            all_samples.append(sample)
    #else:
    #    for sample in all_samples:
    #        query = sample['masked_sentences'][0].replace(base.MASK, '')
    #        sample['query'] = query
    #        #print(f'query={query}')
    #        docs = retrieve_docs(query, ranker, conn, 1)
    #        sample['context'] = docs[0]
            

    # create uuid if not present
    i = 0
    for sample in all_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1

    # shuffle data
    if shuffle_data:
        shuffle(all_samples)

    samples_batches, sentences_batches, ret_msg = batchify(all_samples, args.batch_size)
    logger.info("\n" + ret_msg + "\n")

    # ThreadPool
    num_threads = args.threads
    if num_threads <= 0:
        # use all available threads
        num_threads = multiprocessing.cpu_count()
    pool = ThreadPool(num_threads)
    list_of_results = []

    viz = False
    num_viz = 10
    final_viz = []
    viz_thres = 11
    qamodel.eval().cuda()
    # Defaults from huggingface
    do_lower_case = True
    max_answer_length = 30
    verbose_logging = False
    null_score_diff_threshold = 0.0
    n_best = 20
    max_query_length = 64
    # Training specifics:
    doc_stride = 128
    max_seq_length = 384


    for i in tqdm(range(len(samples_batches))):

        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]
        mymodel_probs_list = []
        predictions_list = []

        examples = read_input_examples(samples_b)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
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
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=len(samples_b))
        all_results = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            #stime = time.time()
            batch = tuple(t.cuda() for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1]
                          }
                inputs['token_type_ids'] = batch[2]  # XLM don't use segment_ids
                example_indices = batch[3]
                outputs = qamodel(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
                all_results.append(result)
            #total_time = time.time() - stime
            #print(total_time)
            #import ipdb
            #ipdb.set_trace()


        predictions = get_predictions(examples, features, all_results, n_best,
            max_answer_length, do_lower_case,
            verbose_logging,
            v2, null_score_diff_threshold, must_choose_answer=must_choose_answer)
        predictions = [predictions[p] for p in predictions]
        predictions_list.extend(predictions)

        torch.cuda.empty_cache()

        original_log_probs_list, token_ids_list, masked_indices_list = model.get_batch_generation(
            sentences_b, logger=logger
        )
        mymodel_probs_list = original_log_probs_list

        #obj_len = 0
        #for obj in gc.get_objects():
        #    try:
        #        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #            print(type(obj), obj.size())
        #            obj_len += 1
        #    except:
        #        pass
        #print(obj_len)

        if vocab_subset is not None:
            # filter log_probs
            filtered_log_probs_list = model.filter_logprobs(
                original_log_probs_list, filter_logprob_indices
            )
        else:
            filtered_log_probs_list = original_log_probs_list

        label_index_list = []
        for sample in samples_b:
            obj_label_id = model.get_id(sample["obj_label"])

            # MAKE SURE THAT obj_label IS IN VOCABULARIES
            if obj_label_id is None:
                raise ValueError(
                    "object label {} not in model vocabulary".format(
                        sample["obj_label"]
                    )
                )
            #elif model.vocab[obj_label_id[0]] != sample["obj_label"]:
            #    raise ValueError(
            #        "object label {} not in model vocabulary".format(
            #            sample["obj_label"]
            #        )
            #    )
            elif vocab_subset is not None and sample["obj_label"] not in vocab_subset:
                raise ValueError(
                    "object label {} not in vocab subset".format(sample["obj_label"])
                )

            label_index_list.append(obj_label_id)

        arguments = [
            {
                "mymodel_probs": mymodel_probs,
                "original_log_probs": original_log_probs,
                "filtered_log_probs": filtered_log_probs,
                "target": sample["reconstructed_word"],
                "prediction": pred,
                "token_ids": token_ids,
                "vocab": model.vocab,
                "label_index": label_index[0] if len(label_index) > 0 else 0,
                "masked_indices": masked_indices,
                "interactive": args.interactive,
                "index_list": index_list,
                "sample": sample,
            }
            for mymodel_probs, original_log_probs, filtered_log_probs, token_ids, masked_indices, label_index, sample, pred in zip(
                mymodel_probs_list,
                original_log_probs_list,
                filtered_log_probs_list,
                token_ids_list,
                masked_indices_list,
                label_index_list,
                samples_b,
                predictions_list,
            )
        ]

        # single thread for debug
        # for isx,a in enumerate(arguments):
        #     print(samples_b[isx])
        #     run_thread(a)

        # multithread
        res = pool.map(run_thread, arguments)


        for idx, result in enumerate(res):

            result_masked_topk, sample_MRR, sample_P, sample_perplexity, msg, sample_em, sample_f1, sample_is_error, sample_no_overlap, sample_larger_by_1, sample_larger_by_2, sample_larger_by_3, sample_larger_by_4, sample_larger_by_5_or_more = result

            logger.info("\n" + msg + "\n")

            sample = samples_b[idx]

            element = {}
            element["sample"] = sample
            element["uuid"] = sample["uuid"]
            element["token_ids"] = token_ids_list[idx]
            element["masked_indices"] = masked_indices_list[idx]
            element["label_index"] = label_index_list[idx]
            element["masked_topk"] = result_masked_topk
            element["sample_MRR"] = sample_MRR
            element["sample_Precision"] = sample_P
            element["sample_perplexity"] = sample_perplexity
            element["sample_Precision1"] = result_masked_topk["P_AT_1"]
            element['sample_em'] = sample_em
            element['sample_f1'] = sample_f1

            # print()
            # print("idx: {}".format(idx))
            # print("masked_entity: {}".format(result_masked_topk['masked_entity']))
            # for yi in range(10):
            #     print("\t{} {}".format(yi,result_masked_topk['topk'][yi]))
            # print("masked_indices_list: {}".format(masked_indices_list[idx]))
            # print("sample_MRR: {}".format(sample_MRR))
            # print("sample_P: {}".format(sample_P))
            # print("sample: {}".format(sample))
            # print()

            MRR += sample_MRR
            Precision += sample_P
            Precision1 += element["sample_Precision1"]
            is_error += sample_is_error
            no_overlap += sample_no_overlap
            larger_by_1 += sample_larger_by_1
            larger_by_2 += sample_larger_by_2
            larger_by_3 += sample_larger_by_3
            larger_by_4 += sample_larger_by_4
            larger_by_5_or_more += sample_larger_by_5_or_more
            EM += sample_em
            F1 += sample_f1

            # the judgment of the annotators recording whether they are
            # evidence in the sentence that indicates a relation between two entities.
            num_yes = 0
            num_no = 0

            if "judgments" in sample:
                # only for Google-RE
                for x in sample["judgments"]:
                    if x["judgment"] == "yes":
                        num_yes += 1
                    else:
                        num_no += 1
                if num_no >= num_yes:
                    samples_with_negative_judgement += 1
                    element["judgement"] = "negative"
                    MRR_negative += sample_MRR
                    Precision_negative += sample_P
                else:
                    samples_with_positive_judgement += 1
                    element["judgement"] = "positive"
                    MRR_positive += sample_MRR
                    Precision_positivie += sample_P

            list_of_results.append(element)

    if viz:
        with open('viz.pkl', 'wb') as wf:
            pickle.dump(final_viz, wf)

    pool.close()
    pool.join()

    # stats
    # Mean reciprocal rank
    MRR /= len(list_of_results)

    # Precision
    Precision /= len(list_of_results)
    Precision1 /= len(list_of_results)

    EM /= len(list_of_results)
    F1 /= len(list_of_results)

    msg = "all_samples: {}\n".format(len(all_samples))
    msg += "list_of_results: {}\n".format(len(list_of_results))
    msg += "global MRR: {}\n".format(MRR)
    msg += "global Precision at 10: {}\n".format(Precision)
    msg += "global Precision at 1: {}\n".format(Precision1)
    msg += "global EM {}\n".format(EM)
    msg += "global F1: {}\n".format(F1)

    if samples_with_negative_judgement > 0 and samples_with_positive_judgement > 0:
        # Google-RE specific
        MRR_negative /= samples_with_negative_judgement
        MRR_positive /= samples_with_positive_judgement
        Precision_negative /= samples_with_negative_judgement
        Precision_positivie /= samples_with_positive_judgement
        msg += "samples_with_negative_judgement: {}\n".format(
            samples_with_negative_judgement
        )
        msg += "samples_with_positive_judgement: {}\n".format(
            samples_with_positive_judgement
        )
        msg += "MRR_negative: {}\n".format(MRR_negative)
        msg += "MRR_positive: {}\n".format(MRR_positive)
        msg += "Precision_negative: {}\n".format(Precision_negative)
        msg += "Precision_positivie: {}\n".format(Precision_positivie)

    logger.info("\n" + msg + "\n")
    print("\n" + msg + "\n")

    # dump pickle with the result of the experiment
    all_results = dict(
        list_of_results=list_of_results, global_MRR=MRR, global_P_at_10=Precision
    )
    with open("{}/result.pkl".format(log_directory), "wb") as f:
        pickle.dump(all_results, f)

    return Precision1, Precision, MRR, EM, F1, is_error, no_overlap, larger_by_1, larger_by_2, larger_by_3, larger_by_4, larger_by_5_or_more


if __name__ == "__main__":
    parser = options.get_eval_KB_completion_parser()
    args = options.parse_args(parser)
    main(args)

"""
RE-Flex model
"""

from reflex.utils import chunks
from fairseq.models.roberta import RobertaModel, alignment_utils
import torch
import torch.nn.functional as F
from spacy.tokenizer import Tokenizer
from collections import defaultdict
import numpy as np

class Reflex():
    def __init__(self, model_dir, model_name, device, k, spacy_model):
        self.model = RobertaModel.from_pretrained(model_dir, checkpoint_file=model_name)
        self.model.to(device=device)
        self.device = device
        self.bpe = self.model.bpe
        self.task = self.model.task
        self.max_sentence_length = 256
        self.mask = "<mask>" 
        self.start_sentence = "<s>"
        self.period = '.'
        self.k = k
        self.nlp = spacy_model
        self.tokenizer = Tokenizer(self.nlp.vocab)

    def predict(self, batch):
        _, mask_idxs, context_lengths, spacy_tokenss, alignments = batch
        batched_p_B_theta = self.compute_p_B_theta(batch)
        predictions = []
        for ind, unnormalized_p_B_theta in enumerate(batched_p_B_theta):
            p_B_theta = F.softmax(unnormalized_p_B_theta, dim=0)
            p_B, fixed_vals = torch.topk(p_B_theta.squeeze(), k=self.k, dim=0)
            prepared_fixed_batch = self.prepare_fixed_batch(batch, ind, fixed_vals)
            features = self.compute_batch_features(prepared_fixed_batch)
            context_length = context_lengths[ind]
            alignment = alignments[ind]
            spacy_tokens = spacy_tokenss[ind]
            # Filter punctuation and stop words from D
            valid_inds = []
            for st_ind, tok in enumerate(spacy_tokens):
                #if self.nlp.vocab[tok].is_stop or self.nlp.vocab[tok].is_punct:
                #    continue
                #if self.nlp.vocab[tok].is_punct:
                #    continue
                valid_inds.append(st_ind)

            mask_idx = mask_idxs[ind]
            p_D = torch.zeros((len(valid_inds)), dtype=torch.float).to(device=self.device)
            valid_inds = torch.from_numpy(np.array(valid_inds)).long().to(device=self.device)
            for p_ind, p_b in enumerate(p_B):
                feat = features[p_ind].squeeze()
                context_features = feat[:context_length, :] # num_spacy_tokens x C
                context_features = self.align_features_to_words(context_features, alignment)[1:, :] # chop off the start sentence token
                #import ipdb
                #ipdb.set_trace()
                context_features = torch.index_select(context_features, dim=0, index=valid_inds)
                mask_features = feat[mask_idx].repeat(context_features.shape[0], 1)
                p_D_given_b = F.cosine_similarity(context_features, mask_features, dim=1)
                p_D_given_b = F.softmax(p_D_given_b, dim=0)
                # Marginalize
                p_D += p_D_given_b * p_b
            map_ind = torch.argmax(p_D, dim=0)
            predictions.append(spacy_tokens[map_ind])
        return predictions

    def prepare_fixed_batch(self, batch, batch_ind, fixed_vals):
        tens, mask_idxs, _, _, _ = batch
        sample = tens[batch_ind]
        mask_idx = mask_idxs[batch_ind][0][0]
        fixed_batch = []
        for fv in fixed_vals:
            fixed_sample = sample.clone().squeeze()
            fixed_sample[mask_idx] = fv
            fixed_batch.append(fixed_sample)
        return torch.stack(fixed_batch)

    def align_bpe_to_words(self, bpe_tokens, other_tokens):
        # subroutine of fairseq.models.roberta.alignment_utils.align_bpe_to_words
        # create alignment from every word to a list of BPE tokens

        assert ''.join(bpe_tokens) == ''.join(other_tokens)
        alignment = []
        bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))
        j, bpe_tok = next(bpe_toks)
        for other_tok in other_tokens:
            bpe_indices = []
            while True:
                if other_tok.startswith(bpe_tok):
                    bpe_indices.append(j)
                    other_tok = other_tok[len(bpe_tok):]
                    try:
                        j, bpe_tok = next(bpe_toks)
                    except StopIteration:
                        j, bpe_tok = None, None
                elif bpe_tok.startswith(other_tok):
                    # other_tok spans multiple BPE tokens
                    bpe_indices.append(j)
                    bpe_tok = bpe_tok[len(other_tok):]
                    other_tok = ''
                else:
                    raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
                if other_tok == '':
                    break
            assert len(bpe_indices) > 0
            alignment.append(bpe_indices)
        assert len(alignment) == len(other_tokens)

        return alignment

    def align_features_to_words(self, features, alignment):
        # slight modification of fairseq.models.roberta.alignment_utils.align_features_to_words to compute average
        # instead of sum
        assert features.dim() == 2
    
        bpe_counts = defaultdict(int)
        for bpe_indices in alignment:
            for j in bpe_indices:
                bpe_counts[j] = len(bpe_indices)
        denom = features.new([bpe_counts[j] if bpe_counts[j] > 0 else 1 for j in range(len(features))])
        weighted_features = features / denom.unsqueeze(-1)
    
        output = [weighted_features[0]]
        largest_j = -1
        for bpe_indices in alignment:
            output.append(weighted_features[bpe_indices].sum(dim=0))
        output = torch.stack(output)
        #assert torch.all(torch.abs(output.sum(dim=0) - features.sum(dim=0)) < 1e-3)
        return output

    def get_context_alignment(self, context):
        context_spans = self.bpe.encode(context)
        text_spans_context = f'{self.start_sentence} {context_spans}'
        context_encoded = self.task.source_dictionary.encode_line(text_spans_context, append_eos=False)
        # Sometimes the source dictionary decodes the unicode differently than spacy
        # so we use the decoded context instead of the original context
        decoded = [self.get_bpe_val(i) for i in context_encoded[1:]]
        decoded_context = ''.join(decoded)
        decoded = [i.strip() for i in decoded]
        spacy_tokens = [t.text.strip() for t in self.tokenizer(decoded_context)]
        if ''.join(decoded) != ''.join(spacy_tokens):
            import ipdb
            ipdb.set_trace()
        alignment = self.align_bpe_to_words(decoded, spacy_tokens)

        return len(context_encoded), spacy_tokens, alignment

    def encode_context(self, context, text_spans_bpe):
        context_spans = self.bpe.encode(context)
        text_spans_context = f'{self.start_sentence} {context_spans} {text_spans_bpe}'
        context_encoded = self.task.source_dictionary.encode_line(text_spans_context, append_eos=True)
        context_length, spacy_tokens, alignment = self.get_context_alignment(context)

        return context_encoded, context_length, spacy_tokens, alignment

    def process_context(self, context, text_spans_bpe, t):
        if context is None:
            text_spans_bpe = f'{self.start_sentence} {text_spans_bpe}'
            encoded = self.task.source_dictionary.encode_line(
                text_spans_bpe, append_eos=True
            )
            return encoded
        context = context.rstrip()
        context_encoded, context_length, spacy_tokens, alignment = self.encode_context(context, text_spans_bpe)
        while len(context_encoded) > t:
            # For now, we prune the context
            context = context[:-10]
            context_encoded, context_length, spacy_tokens, alignment = self.encode_context(context, text_spans_bpe)
        return context_encoded, context_length, spacy_tokens, alignment

    def get_mask(self):
        return self.task.source_dictionary.index(self.mask)

    def batch(self, samples, bsz):
        encoded_list = []
        for s in samples:
            sample = s.template.replace('[X]', s.head)
            sample = sample.replace('[Y]', self.mask)
            text_spans = sample.split(self.mask)
            text_spans_bpe = f' {self.mask} '.join([self.bpe.encode(ts.rstrip()) for ts in text_spans])
            encoded, context_length, spacy_tokens, alignment = self.process_context(s.context, text_spans_bpe, 500)
            masked_idx = (encoded == self.get_mask()).nonzero().numpy()
            encoded_list.append(((encoded, masked_idx, context_length, spacy_tokens, alignment), s))
        # sort by length of encoded
        encoded_list.sort(key=lambda x: len(x[0][0]))
        # Since we reordered the samples list, we need the original sample list reordered as well for computing ground truth
        encoded_list, samples = zip(*encoded_list)
        batches = []
        for batch in chunks(encoded_list, bsz):
            max_len = len(max(batch, key=lambda x: len(x[0]))[0])
            # Pad the batch according to the max length in the sequence
            encs = []
            idxs = []
            context_lengths = []
            alignments = []
            spacy_tokenss = []
            for encoded, masked_idx, context_length, spacy_tokens, alignment in batch:
                if len(encoded) < max_len:
                    pad_len = max_len - len(encoded)
                    pad = torch.full([pad_len], self.task.source_dictionary.pad(), dtype=torch.int)
                    encoded = torch.cat([encoded, pad])
                encs.append(encoded)
                idxs.append(masked_idx)
                context_lengths.append(context_length)
                alignments.append(alignment)
                spacy_tokenss.append(spacy_tokens)
            batches.append((torch.stack(encs), idxs, context_lengths, spacy_tokenss, alignments))
        return batches, samples

    def get_bpe_val(self, ind):
        bpe = self.task.source_dictionary[ind]
        val = self.bpe.decode(bpe)
        return val

    def compute_batch_features(self, batch):
        batch = batch.long().to(device=self.device)
        with torch.no_grad():
            features = self.model.model.extract_features(batch, return_all_hiddens=True)[1]['inner_states']
        features = torch.stack([f.transpose(0, 1) for f in features], dim=3) # B x T x F x L
        features = features.view(features.shape[0], features.shape[1], -1) # Flatten the layer features for each token: B x T x C
        return features


    def compute_p_B_theta(self, batch):
        tens, idxs, _, _, _ = batch
        with torch.no_grad():
            self.model.eval()
            self.model.model.eval()
            log_probs, extra = self.model.model(
                tens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )

        results = []
        for ind, log_prob in enumerate(log_probs):
            masked_idx = idxs[ind]
            log_prob = log_prob.squeeze()
            mask_probs = log_prob[masked_idx].squeeze()
            results.append(mask_probs)
        return results


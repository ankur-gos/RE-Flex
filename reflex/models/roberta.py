"""
Roberta model
"""

from reflex.utils import chunks, get_bpe_val
from fairseq.models.roberta import RobertaModel
import torch

class Roberta():
    def __init__(self, model_dir, model_name, device):
        self.model = RobertaModel.from_pretrained(model_dir, checkpoint_file=model_name)
        self.model.to(device=device)
        self.device = device
        self.bpe = self.model.bpe
        self.task = self.model.task
        self.max_sentence_length = 256
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0)
        self.mask = "<mask>" 
        self.start_sentence = "<s>"
        self.period = '.'

    def update_batch(self, batch, new_toks):
        tens, mask_idxs = batch
        # Update the batch
        batch_list = []
        for ind, tok in enumerate(new_toks):
            mask_idx = mask_idxs[ind][0][0]
            t = tens[ind].squeeze()
            right_len = t.shape[0] - mask_idx - 1
            l, r = torch.split(t, [mask_idx+1, right_len], dim=0)
            l[mask_idx] = tok
            new_t = torch.cat([l, torch.IntTensor([self.get_mask()]), r])
            batch_list.append(new_t)
            mask_idxs[ind] += 1
        new_batch = torch.stack(batch_list)
        return (new_batch, mask_idxs)

    def end_seq_condition(self, token):
        return token == "</s>" or token == self.period

    def decode_lm(self, batch, cap):
        i = 0
        # maintain batch state in this list. Finish decoding when all states have been flipped
        eos_state = [False] * len(batch[1])
        while True:
            if i == 0:
                results = self.decode_naive(batch)
                token_list = [[get_bpe_val(r, self.task.source_dictionary, self.bpe)] for r in results]
                batch = self.update_batch(batch, results)
                i += 1
            else:
                if False not in eos_state or i > cap:
                    break
                results = self.decode_naive(batch)
                for ind, t in enumerate(token_list):
                    new_tok = get_bpe_val(results[ind], self.task.source_dictionary, self.bpe)
                    if self.end_seq_condition(new_tok):
                        eos_state[ind] = True
                    t.append(new_tok)
                i += 1

        #import ipdb
        #ipdb.set_trace()
        final_strings = []
        for tl in token_list:
            tmp_tl = tl
            for ind, t in enumerate(tl):
                if self.end_seq_condition(tl[ind]):
                    tmp_tl = tl[:ind]
                    break
            final_strings.append(''.join(tmp_tl))
        return final_strings

    def encode_context(self, context, text_spans_bpe):
        context_spans = self.bpe.encode(context)
        text_spans_context = f'{self.start_sentence} {context_spans} {text_spans_bpe}'
        context_encoded = self.task.source_dictionary.encode_line(text_spans_context, append_eos=True)
        return context_encoded

    def process_context(self, context, text_spans_bpe, t):
        if context is None:
            text_spans_bpe = f'{self.start_sentence} {text_spans_bpe}'
            encoded = self.task.source_dictionary.encode_line(
                text_spans_bpe, append_eos=True
            )
            return encoded
        context = context.rstrip()
        context_encoded = self.encode_context(context, text_spans_bpe)
        while len(context_encoded) > t:
            # For now, we prune the context
            context = context[:-10]
            context_encoded = self.encode_context(context, text_spans_bpe)
        return context_encoded

    def get_mask(self):
        return self.task.source_dictionary.index(self.mask)


    def batch(self, samples, bsz):
        encoded_list = []
        for s in samples:
            sample = s.template.replace('[X]', s.head)
            sample = sample.replace('[Y]', self.mask)
            text_spans = sample.split(self.mask)
            text_spans_bpe = f' {self.mask} '.join([self.bpe.encode(ts.rstrip()) for ts in text_spans])
            encoded = self.process_context(s.context, text_spans_bpe, 500)
            masked_idx = (encoded == self.get_mask()).nonzero().numpy()
            encoded_list.append(((encoded, masked_idx), s))
        # sort by length of encoded
        encoded_list.sort(key=lambda x: len(x[0][0]))
        encoded_list, samples = zip(*encoded_list)
        batches = []
        for batch in chunks(encoded_list, bsz):
            max_len = len(max(batch, key=lambda x: len(x[0]))[0])
            # Pad the batch according to the max length in the sequence
            encs = []
            idxs = []
            for encoded, masked_idx in batch:
                if len(encoded) < max_len:
                    pad_len = max_len - len(encoded)
                    pad = torch.full([pad_len], self.task.source_dictionary.pad(), dtype=torch.int)
                    encoded = torch.cat([encoded, pad])
                encs.append(encoded)
                idxs.append(masked_idx)
            batches.append((torch.stack(encs), idxs))
        return batches, samples

    def decode_naive(self, batch):
        tens, idxs = batch
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
            decode_ind = torch.argmax(mask_probs, dim=0)
            results.append(decode_ind)
            
        return results


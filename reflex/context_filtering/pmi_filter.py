import numpy as np
import fasttext
import sqlite3
import random
from scipy.stats import norm
import spacy
import json
from collections import Counter

class WordEmbeddingsPMIFilter:
    def __init__(self, we_model_pth, lambda_val):
        print("Loading fastext word embeddings ... ")
        self.word_emb = fasttext.load_model(we_model_pth)
        self.filter_tokens = ['.', ',', '(', ')', '</s>', '_._', ':', '-', ',', '_..._', '_:_']
        print("Loading spacy model ... ")
        self.nlp = spacy.load('en_core_web_lg')
        self.lambda = lambda_val

    def estimate_pmi(self, sample):
        template = self.nlp(sample.template.strip())
        context = self.nlp(sample.context.strip())
        for t in template:
            t_text = t.text
            max_score = 0
            for c in context:
                c_text = c.text
                score = self.get_cosine_sim_score(self.word_emb[c_text], self.word_emb[t_text])
                if score > max_score:
                    max_score = score
            set_score.append(max_score)
        return np.average(set_score)

    def filter(self, samples):
        ms = np.asarray([self.estimate_pmi(s) for s in samples])
        mean, std = norm.fit(ms)
        lower_bound = mean + self.lambda * std
        zsamples = zip(samples, ms)
        filtered_samples = []
        for sample, m in zsamples:
            if m >= lower_bound:
                filtered_samples.append(sample)
        return filtered_samples

    def get_cosine_sim_score(self, emb1, emb2):
        return np.dot(emb1, emb2)/(np.linalg.norm(emb1) * np.linalg.norm(emb2))


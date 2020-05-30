import numpy as np
import fasttext
import sqlite3
import random
from scipy.stats import norm
import spacy
import json
from collections import Counter
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

class WordEmbeddingsPMIFilter:
    def __init__(self, we_model, spacy_model, lambda_val):
        print("Loading fastext word embeddings ... ")
        self.word_emb = we_model
        self.filter_tokens = ['.', ',', '(', ')', '</s>', '_._', ':', '-', ',', '_..._', '_:_']
        print("Loading spacy model ... ")
        self.nlp = spacy_model
        self.lambda_val = lambda_val
        self.gmm_estimator = GaussianMixture(n_components=2) # Defaults are reasonable

    def estimate_pmi(self, sample):
        template = sample.template.replace('[X]', sample.head)
        template = template.replace('[Y]', '')
        template = self.nlp(template.strip())
        context = self.nlp(sample.context.strip())
        scores = []
        for t in template:
            t_text = t.text
            max_score = 0
            for c in context:
                c_text = c.text
                score = cosine_similarity(self.word_emb[c_text].reshape(1, -1), self.word_emb[t_text].reshape(1, -1)).squeeze()
                if score > max_score:
                    max_score = score
            scores.append(max_score)
        return np.average(scores)

    def cluster(self, samples):
        ms = np.asarray([self.estimate_pmi(s) for s in samples]).reshape(-1, 1)
        est = self.gmm_estimator.fit(ms)
        means = est.means_
        I_c = 0 if means[0] > means[1] else 1
        preds = est.predict(ms)

        zsamples = zip(samples, ms, preds)
        filtered_samples = []
        for sample, m, p in zsamples:
            if p == I_c:
                filtered_samples.append(sample)
        return filtered_samples


    def filter(self, samples):
        ms = np.asarray([self.estimate_pmi(s) for s in samples])
        mean, std = norm.fit(ms)
        lower_bound = mean + self.lambda_val * std
        zsamples = zip(samples, ms)
        filtered_samples = []
        for sample, m in zsamples:
            if m >= lower_bound:
                filtered_samples.append(sample)
        return filtered_samples

    def get_cosine_sim_score(self, emb1, emb2):
        return np.dot(emb1, emb2)/(np.linalg.norm(emb1) * np.linalg.norm(emb2))


import numpy as np
from scipy.stats import norm
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import fasttext


class WordEmbeddingsPMIFilter:
    def __init__(self, we_model, spacy_model, lambda_val):
        print("Loading fastext word embeddings ... ")
        if isinstance(we_model, str):
            self.word_emb = fasttext.load_model(we_model)
        else:
            self.word_emb = we_model
        if isinstance(spacy_model, str):
            print("Loading spacy model ... ")
            self.nlp = spacy.load(spacy_model)
        else:
            self.nlp = spacy_model
        self.lambda_val = lambda_val

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


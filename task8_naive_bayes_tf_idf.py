import numpy as np
import re
import math
from collections import Counter, defaultdict
import random

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior = {}
        self.word_log_prob = {}
        self.classes = set()
        self.vocabulary = set()

    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def _compute_tf(self, document):
        return Counter(document)

    def _compute_idf(self, documents):
        N = len(documents)
        df = defaultdict(int)
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1
        idf = {term: math.log((N + 1) / (df[term] + 1)) + 1 for term in df}
        return idf

    def _compute_tfidf(self, documents):
        idf = self._compute_idf(documents)
        tfidf_features = []
        for doc in documents:
            tf = self._compute_tf(doc)
            total_terms = len(doc)
            tfidf = {term: (tf[term] / total_terms) * idf[term] for term in tf}
            tfidf_features.append(tfidf)
        return tfidf_features

    def fit(self, X, y):
        tokenized_docs = [self._preprocess_text(doc) for doc in X]
        self.vocabulary = set(term for doc in tokenized_docs for term in doc)
        self.classes = set(y)
        tfidf_docs = self._compute_tfidf(tokenized_docs)

        doc_by_class = defaultdict(list)
        for doc, label in zip(tfidf_docs, y):
            doc_by_class[label].append(doc)

        total_docs = len(y)
        for cls in self.classes:
            self.class_log_prior[cls] = math.log(len(doc_by_class[cls]) / total_docs)

        for cls in self.classes:
            word_sums = defaultdict(float)
            for doc in doc_by_class[cls]:
                for word, val in doc.items():
                    word_sums[word] += val
            total = sum(word_sums.values())
            self.word_log_prob[cls] = {
                word: math.log((word_sums[word] + self.alpha) / (total + self.alpha * len(self.vocabulary)))
                for word in self.vocabulary
            }
        return self

    def predict(self, X):
        return [self._predict_single(doc) for doc in X]

    def _predict_single(self, x):
        tokens = self._preprocess_text(x)
        tf = self._compute_tf(tokens)
        total = len(tokens)
        tf = {term: tf[term] / total for term in tf}
        scores = {}
        for cls in self.classes:
            score = self.class_log_prior[cls]
            for term, freq in tf.items():
                if term in self.word_log_prob[cls]:
                    score += freq * self.word_log_prob[cls][term]
            scores[cls] = score
        return max(scores, key=scores.get)

    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(p == t for p, t in zip(predictions, y))
        return correct / len(y)

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test

# Example usage
imdb_sample = [
    "This movie was fantastic! The acting was incredible and the plot kept me engaged the entire time.",
    "One of the best films I've seen in years. The director did an amazing job with the cinematography.",
    "What a terrible waste of time. The plot made no sense and the acting was wooden at best.",
    "I couldn't wait for this movie to end. Bad dialogue, poor character development, and a predictable storyline."
]

imdb_labels = [1, 1, 0, 0]

X_train, X_test, y_train, y_test = train_test_split(imdb_sample, imdb_labels, test_size=0.5, random_state=42)
nb_classifier = NaiveBayesClassifier(alpha=0.1)
nb_classifier.fit(X_train, y_train)
accuracy = nb_classifier.score(X_test, y_test)
print(f"Accuracy on test set: {accuracy:.2f}")

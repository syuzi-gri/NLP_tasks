# *********** SMS Spam Classification using Markov Chains and Bayes Theorem ***********

import pandas as pd
import numpy as np
import re
import math
from sklearn.model_selection import train_test_split

def clean_text(msg):
    msg = msg.lower()
    msg = re.sub(r'[^a-z\s]', '', msg)
    return msg.split()

data = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)

def train_probs(messages):
    start_counts = {}
    trans_counts = {}
    total_starts = 0

    for msg in messages:
        if not msg:
            continue
        first = msg[0]
        start_counts[first] = start_counts.get(first, 0) + 1
        total_starts += 1

        for i in range(len(msg) - 1):
            pair = (msg[i], msg[i+1])
            trans_counts[pair] = trans_counts.get(pair, 0) + 1

    vocab = set()
    for msg in messages:
        vocab.update(msg)
    vocab_size = len(vocab)

    start_probs = {w: (start_counts.get(w, 0) + 1) / (total_starts + vocab_size) for w in vocab}

    follow_counts = {}
    trans_probs = {}
    for (w1, w2), count in trans_counts.items():
        follow_counts[w1] = follow_counts.get(w1, 0) + count

    for (w1, w2) in trans_counts:
        trans_probs[(w1, w2)] = (trans_counts[(w1, w2)] + 1) / (follow_counts[w1] + vocab_size)

    return start_probs, trans_probs

spam_msgs = [X_train.iloc[i] for i in range(len(X_train)) if y_train.iloc[i] == 'spam']
ham_msgs = [X_train.iloc[i] for i in range(len(X_train)) if y_train.iloc[i] == 'ham']

total = len(spam_msgs) + len(ham_msgs)
P_spam = len(spam_msgs) / total
P_ham = len(ham_msgs) / total

log_spam = math.log(P_spam)
log_ham = math.log(P_ham)

spam_start, spam_trans = train_probs(spam_msgs)
ham_start, ham_trans = train_probs(ham_msgs)

def calc_log_prob(msg, start_probs, trans_probs):
    if not msg:
        return float('-inf')

    prob = math.log(start_probs.get(msg[0], 1e-6))
    for i in range(len(msg) - 1):
        pair = (msg[i], msg[i+1])
        prob += math.log(trans_probs.get(pair, 1e-6))

    return prob

def predict(msg, spam_start, spam_trans, ham_start, ham_trans):
    prob_spam = calc_log_prob(msg, spam_start, spam_trans) + log_spam
    prob_ham = calc_log_prob(msg, ham_start, ham_trans) + log_ham

    return 'spam' if prob_spam > prob_ham else 'ham'

predictions = [predict(msg, spam_start, spam_trans, ham_start, ham_trans) for msg in X_test]

acc = np.mean(np.array(predictions) == np.array(y_test))
print(f"Model accuracy: {acc * 100:.2f}%")

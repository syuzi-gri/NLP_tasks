import pandas as pd
import numpy as np
import re
from collections import defaultdict
from math import log

df = pd.read_csv('spam.csv', encoding='latin1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words

df['tokens'] = df['message'].apply(tokenize)

vocab = set()
for tokens in df['tokens']:
    vocab.update(tokens)
vocab = sorted(list(vocab))

word_counts = {
    'spam': defaultdict(int),
    'ham': defaultdict(int)
}
spam_count = 0
ham_count = 0

for _, row in df.iterrows():
    label = 'spam' if row['label'] == 1 else 'ham'
    if label == 'spam':
        spam_count += 1
    else:
        ham_count += 1
    for word in row['tokens']:
        word_counts[label][word] += 1

total_messages = len(df)
P_spam = spam_count / total_messages
P_ham = ham_count / total_messages

likelihood_spam = {}
likelihood_ham = {}
alpha = 1
vocab_size = len(vocab)
total_spam_words = sum(word_counts['spam'].values())
total_ham_words = sum(word_counts['ham'].values())

for word in vocab:
    likelihood_spam[word] = (word_counts['spam'][word] + alpha) / (total_spam_words + alpha * vocab_size)
    likelihood_ham[word] = (word_counts['ham'][word] + alpha) / (total_ham_words + alpha * vocab_size)

def classify(message):
    tokens = tokenize(message)
    log_spam = log(P_spam)
    log_ham = log(P_ham)
    
    for word in tokens:
        if word in vocab:
            log_spam += log(likelihood_spam[word])
            log_ham += log(likelihood_ham[word])
    
    return 'spam' if log_spam > log_ham else 'ham'

test_message = "Congratulations! You won a prize"
print(f"Message: {test_message}")
print("Classified as:", classify(test_message))

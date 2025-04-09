# *********** BBC News Article Classification ***********

import pandas as pd
import string
import re
from nltk.corpus import stopwords
import nltk
import math

nltk.download('stopwords')

# Load the dataset
df = pd.read_excel('bbc-text.xlsx')
# print(df.head())
# print(df.columns)
# print(df.shape)


# Define stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuation = string.punctuation


# *********** 1	Data Preprocessing *********** 
def preprocess(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(f"[{punctuation}]", "", text)

    # Split
    words = re.findall(r'\b\w+\b', text)

    # Remove stopwords and non-alphabetic tokens
    words = [word for word in words if word.isalpha() and word not in stop_words]

    return words

df['tokens'] = df['text'].apply(preprocess)
print(df.head())



# *********** 2	Implement Bag of Words *********** 
def build_vocabulary(documents):
    vocabulary = set()
    for doc in documents:
        vocabulary.update(doc)
    return sorted(vocabulary)

vocabulary = build_vocabulary(df['tokens'])

def document_to_bow_vector(document, vocabulary):
    bow_vector = [0] * len(vocabulary)
    
    for word in document:
        if word in vocabulary:
            bow_vector[vocabulary.index(word)] += 1
    
    return bow_vector

# df['bow_vector'] = df['tokens'].apply(lambda x: document_to_bow_vector(x, vocabulary))

# print(df[['text', 'bow_vector']].head())




# *********** 3	Implement TF-IDF *********** 
def compute_tf(document):
    tf = {}
    total_terms = len(document)
    for word in document:
        tf[word] = tf.get(word, 0) + 1
    
    for word in tf:
        tf[word] = tf[word] / total_terms

    return tf

df['tf'] = df['tokens'].apply(compute_tf)

def compute_idf(corpus):
    idf = {}
    total_documents = len(corpus)
    
    term_document_count = {}
    for document in corpus:
        unique_terms = set(document)
        for word in unique_terms:
            term_document_count[word] = term_document_count.get(word, 0) + 1

    for word, doc_count in term_document_count.items():
        idf[word] = math.log(total_documents / (1 + doc_count))
    
    return idf

idf = compute_idf(df['tokens'])

def compute_tfidf(tf, idf):
    tfidf = {}
    for word in tf:
        if word in idf:
            tfidf[word] = tf[word] * idf[word]
        else:
            tfidf[word] = 0
    return tfidf

df['tfidf'] = df['tf'].apply(lambda x: compute_tfidf(x, idf))

print(df[['text', 'tfidf']].head())




# *********** 4 Analysis *********** 
def top_tfidf_by_category(df, category_column='category', tfidf_column='tfidf'):
    df_exploded = df.explode(tfidf_column)
    
    for word_dict in df[tfidf_column]:
        for word, value in word_dict.items():
            if not isinstance(value, (int, float)):
                print(f"Non-numeric value found for word: {word}, value: {value}")
                word_dict[word] = 0

    df_exploded = df_exploded.apply(lambda row: pd.Series(row[tfidf_column]), axis=1)
    df_exploded['category'] = df['category']

    category_tfidf = df_exploded.groupby([category_column]).mean(numeric_only=True).reset_index()

    top_words = category_tfidf.melt(id_vars=[category_column], var_name='word', value_name='avg_tfidf')
    top_words = top_words.groupby(category_column, group_keys=False).apply(lambda x: x.nlargest(10, 'avg_tfidf')).reset_index(drop=True)

    return top_words

top_words = top_tfidf_by_category(df)
print(top_words)

def high_tf_low_idf(df, threshold=0.1):
    high_tf_low_idf_words = []
    low_tf_high_idf_words = []
    
    for _, row in df.iterrows():
        for word, tf_value in row['tf'].items():
            idf_value = idf.get(word, 0)
            if tf_value > threshold and idf_value < 0.5: 
                high_tf_low_idf_words.append(word)
            elif tf_value < threshold and idf_value > 1.0:
                low_tf_high_idf_words.append(word)
    
    return high_tf_low_idf_words, low_tf_high_idf_words

high_tf_low_idf_words, low_tf_high_idf_words = high_tf_low_idf(df)
print("High TF, Low IDF words:", high_tf_low_idf_words)
print("Low TF, High IDF words:", low_tf_high_idf_words)
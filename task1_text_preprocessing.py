# *********** Text Preprocessing ***********

import string

text = """Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. 
It's used to analyze text, allowing machines to understand, interpret, and manipulate human language. 
NLP has many real-world applications, including machine translation, sentiment analysis, and chatbots."""


# 1.	Tokenization: Split the text into individual words (tokens).
tokens = text.split()
print("***** 1 Tokenization *****\n", tokens)


# 2.	Lowercasing: Convert all tokens to lowercase.
tokens_low = [i.lower() for i in tokens]
print("\n***** 2 Lowercasing *****\n", tokens_low)


#3.	Punctuation Removal: Remove all punctuation marks from the tokens.
tokens_rm_punct = [i.strip(string.punctuation) for i in tokens_low]
print("\n***** 3 Punctuation Removal *****\n", tokens_rm_punct)


#4.	Stop Word Removal: Remove common stop words (e.g., "the", "is", "and").
stop_words = ["the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are", "with"]

tokens_rm_stop_words = [word for word in tokens_rm_punct if word not in stop_words]
print("\n***** 4 Stop words removal *****\n", tokens_rm_stop_words)


#5.	Stemming: Reduce words to their root form using a simple algorithm.
def stemming(word):
    for suffix in ['ing', 'ed', 'es', 's']:
        if word.endswith(suffix) and len(word) > len(suffix)+1:
            return word[:-len(suffix)]
    return word

stemmed_tokens = [stemming(word) for word in tokens_rm_stop_words]
print("\n***** 5 Stemming *****\n", stemmed_tokens)


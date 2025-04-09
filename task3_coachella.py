# *********** Text Preprocessing using Coachella dataset ***********

from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import pandas as pd
import re

import nltk
nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt', download_dir=nltk.data.find('tokenizers').path)



# Load data
df = pd.read_csv('task1/coachella.csv', encoding='latin1')
# print(df[['text']].head())

# Hashtag and email
def extract_hashtags(text):
    return re.findall(r"#\w+", str(text))

def extract_emails(text):
    return re.findall(r"\b[\w.-]+?@\w+?\.\w+?\b", str(text))

df['hashtags'] = df['text'].apply(extract_hashtags)
df['emails'] = df['text'].apply(extract_emails)



stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = re.sub(r"@\w+", "", text)                       
    text = re.sub(r"http[s]?://\S+", "", text)           
    text = text.encode('ascii', 'ignore').decode('ascii') 
    text = text.lower()                                  
    text = re.sub(r"\d+", "", text)                     
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split() 
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)


df['cleaned_text'] = df['text'].apply(clean_text)
print(df['cleaned_text'].head())



# for i, row in df.iterrows():
#     try:
#         clean_text(row['text'])
#     except Exception as e:
#         print(f"Error at row {i}: {e}")
#         break
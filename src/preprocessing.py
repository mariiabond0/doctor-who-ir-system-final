import nltk
from nltk.stem import PorterStemmer

# from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# from nltk.stem import PorterStemmer
import string
import pandas as pd

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def preprocess_text(text):
    """Lowercase text, remove punctuation and stopwords, apply stemming"""
    if pd.isna(text) or not text:
        return []
    #    text = text.lower()
    #    tokens = word_tokenize(text)
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [t for t in tokens if t.strip() and t not in string.punctuation and t not in STOPWORDS]
    tokens = [STEMMER.stem(t) for t in tokens]
    return tokens

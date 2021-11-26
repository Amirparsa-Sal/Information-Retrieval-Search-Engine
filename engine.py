from __future__ import unicode_literals
from codecs import register
from hazm import Normalizer,Stemmer,word_tokenize
import string

from hazm.Lemmatizer import Lemmatizer

normalizer = Normalizer()
stemmer = Stemmer()
lemmatizer = Lemmatizer()
index = dict()

def perform_linguistic_preprocessing(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = normalizer.normalize(text)
    token_list = word_tokenize(text)
    token_list = list(map(lambda word: lemmatizer.lemmatize(word), token_list))
    return token_list
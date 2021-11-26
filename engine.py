from __future__ import unicode_literals
from hazm import Normalizer, word_tokenize
import string
import openpyxl
from parsivar import FindStems
import re

normalizer = Normalizer()
stemmer = FindStems()
index = dict() #token -> [freq, [doc_id, freq, pos1, pos2, ...], [doc_id, freq, pos1, pos2, ...], ...]
EXCEL_FILE_NAME = 'data.xlsx'
LINK_REGEX = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([ا-یa-zA-Z0-9\.\&\/\?\:@\-_=# ])*"


def perform_linguistic_preprocessing(text):
    text = re.sub(LINK_REGEX, '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = normalizer.normalize(text)
    token_list = word_tokenize(text)
    token_list = list(map(lambda word: stemmer.convert_to_stem(word), token_list))
    return token_list

wb = openpyxl.load_workbook(EXCEL_FILE_NAME)
sheet = wb.active


# content = sheet.cell(row=2, column=1).value
# print(perform_linguistic_preprocessing(content))
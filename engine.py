from __future__ import unicode_literals
from hazm import Normalizer, word_tokenize, stopwords_list
import string
from hazm.utils import words_list
import openpyxl
from parsivar import FindStems
import re
from collections import OrderedDict
import json
import os
import matplotlib.pyplot as plt
import math

stop_words = stopwords_list()
stop_words.extend(['،','؛','»','«'])
normalizer = Normalizer()
stemmer = FindStems()
index = dict() #token -> [freq, {doc_id1: [freq, pos1, pos2, ...], doc_id2: [freq, pos1, pos2, ...], ...}]
EXCEL_FILE_NAME = 'data.xlsx'
LINK_REGEX = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([ا-یa-zA-Z0-9\.\&\/\?\:@\-_=# ])*"


def perform_linguistic_preprocessing(text, delete_stop_words=True):
    text = re.sub(LINK_REGEX, '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = normalizer.normalize(text)
    token_list = word_tokenize(text)
    token_positions = []
    if delete_stop_words:
        for i,token in enumerate(token_list):
            if token not in stop_words:
                token_positions.append([token, i+1])
    else:
        for i,token in enumerate(token_list):
                token_positions.append([token, i+1])

    return list(map(lambda token: (stemmer.convert_to_stem(token[0]), token[1]), token_positions))

wb = openpyxl.load_workbook(EXCEL_FILE_NAME)
sheet = wb.active

def create_index(index, delete_stop_words=True):
    for i in range(2, sheet.max_row):
        content = sheet.cell(row=i, column=1).value
        token_list = perform_linguistic_preprocessing(content, delete_stop_words=delete_stop_words)
        for token in token_list:
            word = token[0]
            pos = token[1]
            if word in index:
                index[word][0] += 1
                if (i-1) in index[word][1]:
                    index[word][1][i-1][0] += 1
                    index[word][1][i-1][1].append(pos)
                else:
                    index[word][1][i-1] = [1, [pos]]
            else:
                ordered_dict = OrderedDict()
                ordered_dict[i-1] = [1, [pos]]
                index[word] = [1, ordered_dict]

    index = OrderedDict(sorted(index.items(), key=lambda x: x[0], reverse=False))
    json.dump(index, open('index.json', 'w'))

def read_index():
    value = json.loads(open('index.json', 'r').read())
    return OrderedDict(value.items())

def single_word_query(word):
    if word in index:
        news = []
        for doc_id in index[word][1]:
            news.append(doc_id)
        return news
    else:
        return []

def has_positions(postings1, doc_id1, postings2, doc_id2):
    pos_pointer1 = 0
    pos_pointer2 = 0
    pos_list1 = postings1[1][doc_id1][1]
    pos_list2 = postings2[1][doc_id1][1]
    while pos_pointer1 < len(pos_list1) and pos_pointer2 < len(pos_list2):
        pos1 = pos_list1[pos_pointer1]
        pos2 = pos_list2[pos_pointer2]
        if pos1 == pos2 - 1:
            return True
        elif pos1 < pos2 - 1:
            pos_pointer1 += 1
        else:
            pos_pointer2 += 1
    return False

def intersect(list1, list2):
    i = 0
    j = 0
    result = []
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1
    return result

def combine_results(results1, results2):
    for r2 in results2:
        if r2 not in results1:
            results1.append(r2)

def double_word_query(word1, word2):
    postings1 = index.get(word1, None)
    postings2 = index.get(word2, None)
    if postings1 is not None and postings2 is not None:
        news = []
        doc_pointer1 = 0
        doc_pointer2 = 0
        doc_id_list1 = list(postings1[1].keys())
        doc_id_list2 = list(postings2[1].keys())
        while doc_pointer1 < len(doc_id_list1) and doc_pointer2 < len(doc_id_list2):
            doc_id1 = doc_id_list1[doc_pointer1]
            doc_id2 = doc_id_list2[doc_pointer2]
            if doc_id1 == doc_id2:
                if has_positions(postings1, doc_id1, postings2, doc_id2):
                    news.append(doc_id1)
                doc_pointer2 += 1
                doc_pointer1 += 1
            elif int(doc_id1) < int(doc_id2):
                doc_pointer1 += 1
            else:
                doc_pointer2 += 1
        return news
    else:
        return []

def multiple_word_query(query):
    if len(query) == 1:
        return single_word_query(query[0])
    elif len(query) == 2:
        return double_word_query(query[0], query[1])
    else:
        news = []
        for i in range(len(query), 1, -1): # Making different lengths
            for j in range(0, len(query) - i + 1): # Making sublists of query with the given length
                search_list = query[j:j+i]
                result = None
                for k in range(len(search_list) - 1): # Intersect each double word query within the sublist
                    if result is None:
                        result = double_word_query(search_list[k], search_list[k+1])
                    else:
                        result = intersect(result, double_word_query(search_list[k], search_list[k+1]))
                    if result is None:
                        break
                if len(result) > 0:
                    combine_results(news, result)
        for i in range(len(query)):
            combine_results(news, single_word_query(query[i]))

        return news

def plot_zipf_law(index):
    index = OrderedDict(sorted(index.items(), key=lambda x: x[1][0], reverse=True))
    word_list = list(index.keys())
    count_multiply_rank = []
    count = []
    for i in range(len(word_list)):
        count_multiply_rank.append(math.log10(index[word_list[i]][0]))
        count.append(index[word_list[i]][0])
    ranks = list(map(lambda x: math.log10(x), range(1, len(word_list)+1)))
    plt.plot(ranks, count_multiply_rank)
    # plt.plot(list(range(1, len(word_list)+1)), count)
    plt.show()

if not os.path.exists('index.json'):
    create_index(index, delete_stop_words=False)

index = read_index()
print(list(index.items())[0])
index = OrderedDict(sorted(index.items(), key=lambda x: x[1][0], reverse=True))
print(list(index.items())[0][1][0], list(index.items())[0][0])
print(list(index.items())[1][1][0])
print(list(index.items())[2][1][0])
print(list(index.items())[-1][1][0], list(index.items())[-1][0])
print(index[list(index.keys())[0]][0])
plot_zipf_law(index)
while True:
    query = input('Enter your query: ')
    query = list(map(lambda token: token[0], perform_linguistic_preprocessing(query)))
    if len(query) != 0:
        news = multiple_word_query(query)
        if news is None:
            print('No news found')
        else:
            print('News found:')
            news = list(map(lambda doc_id: (sheet.cell(row=int(doc_id)+1, column=3).value,doc_id), news))
            for i,new in enumerate(news):
                print(new[1] + ': ' + new[0])
    else:
        print('No news found')
        
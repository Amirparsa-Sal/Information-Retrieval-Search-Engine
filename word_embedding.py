from engine import perform_linguistic_preprocessing, read_dic_from_file
import openpyxl
import pickle
import math
import numpy as np
from numpy.linalg import norm
from gensim.models import Word2Vec
import os
import multiprocessing
import time 

EXCEL_FILE_NAME = 'data.xlsx'
wb = openpyxl.load_workbook(EXCEL_FILE_NAME)
sheet = wb.active

def create_tokens_list(sheet, delete_stop_words=True):
    doc_token_list = []
    # Looping over the excel file
    for i in range(2, sheet.max_row + 1):
        content = sheet.cell(row=i, column=1).value
        # Performing linguistic preprocessing on the content
        token_list = perform_linguistic_preprocessing(content, delete_stop_words=delete_stop_words)
        only_tokens_list = [item[0] for item in token_list]
        doc_token_list.append(only_tokens_list)
    pickle.dump(doc_token_list, open("token_list.obj", "wb"))
    return doc_token_list

def create_token_list_with_count(sheet, delete_stop_words=True):
    doc_token_list_with_counts = []
    # Looping over the excel file
    for i in range(2, sheet.max_row + 1):
        content = sheet.cell(row=i, column=1).value
        # Performing linguistic preprocessing on the content
        token_list = perform_linguistic_preprocessing(content, delete_stop_words=delete_stop_words)
        token_list = [item[0] for item in token_list]
        token_list_with_count = dict(zip(token_list, [token_list.count(item) for item in token_list]))
        doc_token_list_with_counts.append(token_list_with_count)
    pickle.dump(doc_token_list_with_counts, open("token_list_count.obj", "wb"))
    return doc_token_list_with_counts


def create_tf_idf_list(doc_number):
    index = read_dic_from_file('index.json')
    index = {key: index[key][2] for key in index}
    doc_token_list_with_counts = create_token_list_with_count(sheet)
    for dic in doc_token_list_with_counts:
        for key in dic:
            dic[key] = (1 + math.log10(dic[key])) * math.log10(doc_number / index[key])
    pickle.dump(doc_token_list_with_counts, open("tf_idf_dic.obj", "wb"))
    return doc_token_list_with_counts

def create_docs_matrix(model, docs_tf_id_list):
    matrix = np.matrix(np.zeros((300, len(docs_tf_id_list))))
    for i in range(len(docs_tf_id_list)):
        vector = np.zeros((300))
        weight_sum = 0
        for key in docs_tf_id_list[i]:
            vector += model.wv[key] * docs_tf_id_list[i][key]
            weight_sum += docs_tf_id_list[i][key]
        if weight_sum != 0:
            vector /= weight_sum
        for j in range(300):
            matrix[j, i] = vector[j]
    return matrix

def similarity(matrix, doc_1, doc_2):
    score = np.dot(matrix[:, doc_1 - 1].T, matrix[:, doc_2 - 1]) / (norm(matrix[:, doc_1 - 1]) * norm(matrix[:, doc_2 - 1]))
    return (score + 1) / 2

def search(model, matrix, query, doc_number):
    index = read_dic_from_file('index.json')
    index = {key: index[key][2] for key in index}
    # Calculate tf-idf for the query
    query_tf_idf_dic = dict(zip(query, [query.count(item) for item in query]))
    for key in query_tf_idf_dic:
        query_tf_idf_dic[key] = (1 + math.log10(query_tf_idf_dic[key])) * math.log10(doc_number / index[key])

    query_vector = np.zeros((300))
    weight_sum = 0
    for key in query:
        if key in model.wv:
            query_vector += model.wv[key] * query_tf_idf_dic[key]
            weight_sum += query_tf_idf_dic[key]
    if weight_sum != 0:
        query_vector /= weight_sum
    scores = [0 for i in range(matrix.shape[1])]
    for i in range(matrix.shape[1]):
        scores[i] += np.dot(query_vector.T, matrix[:, i]) / (norm(matrix[:, i]) * norm(query_vector))
    # Sort the scores in descending order
    scores = sorted(enumerate(scores), key = lambda x: x[1], reverse=True)
    return [str(score[0] + 1) for score in scores[:min(10, len(scores))]]

if __name__ == '__main__':
    doc_token_list = []
    if not os.path.exists("token_list.obj"):
        print("Creating token list")
        doc_token_list = create_tokens_list(sheet)
    else:
        doc_token_list = pickle.load(open("token_list.obj", "rb"))

    model = None
    if not os.path.exists("isna_news.model"):
        model = Word2Vec(min_count = 1,
                            window = 5,
                            vector_size = 300,
                            alpha = 0.03,
                            workers = multiprocessing.cpu_count() - 1)
        model.build_vocab(doc_token_list)
        print('Start Training model...')
        start = time.time()
        model.train(doc_token_list, total_examples = model.corpus_count, epochs = 50)
        end = time.time()
        print(f'Completed in {(end - start)} s')
        model.save('isna_news.model')
        print(f'Model saved')
    else:
        print("Loading model")
        model = Word2Vec.load('isna_news.model')

    del doc_token_list

    tf_idf_list = []
    if not os.path.exists("tf_idf_dic.obj"):
        print("Creating tf-idf list")
        tf_idf_list = create_tf_idf_list(len(doc_token_list))
    else:
        tf_idf_list = pickle.load(open("tf_idf_dic.obj", "rb"))

    # print('Creating matrix...')
    matrix = create_docs_matrix(model, tf_idf_list)
    # print('Calculating similarities...')
    # print(similarity(matrix, 403, 690))
    # print(similarity(matrix, 403, 1))
    # print(similarity(matrix, 403, 1009))
    # print(similarity(matrix, 403, 1515))
    # print(similarity(matrix, 403, 124)) 
    while True:
        query = input('Enter your query: ')
        query = list(map(lambda token: token[0], perform_linguistic_preprocessing(query)))
        if len(query) != 0:
            news = search(model, matrix, query, sheet.max_row - 1)
            if news is None:
                print('No news found')
            else:
                print('News found:')
                news = list(map(lambda doc_id: (sheet.cell(row=int(doc_id)+1, column=3).value,doc_id), news))
                for i,new in enumerate(news):
                    print(new[1] + ': ' + new[0])
        else:
            print('No news found')
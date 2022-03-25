import os
import re
import pymorphy2
import nltk
from nltk.corpus import stopwords
from string import punctuation
import math


# получаем номер документа из его названия
def get_file_number(filename_incoming):
    file_number_res = ""
    for char in filename_incoming:
        if char.isdigit():
            file_number_res = file_number_res + char
    return int(file_number_res)


# достаем токены (уникальные слова) из файла
def tokenize_file(path):
    russian_stopwords = stopwords.words("russian")
    token_pattern = "^[а-яА-Я]+$"

    with open(path, encoding='cp1251', mode='r') as f_page:
        text = f_page.read()

    file_tokens = nltk.word_tokenize(text.lower())
    # исключаем знаки пунктуации и стоп-слова
    file_tokens = [file_token for file_token in file_tokens
                   if file_token not in russian_stopwords and file_token not in punctuation]
    # "уникализация"
    file_tokens = set(file_tokens)
    # фильтрация по заданному паттерну
    filtered_file_tokens = [file_token for file_token in file_tokens
                            if re.match(token_pattern, file_token)]
    return filtered_file_tokens


# вычисляем tf-значения для каждого слова из словаря word_dict
# file_tokens - токены конкретного файла
def compute_tf_for_file(word_dict, file_tokens):
    tf_l = {}
    sum_nk = len(file_tokens)
    for word_l, count_l in word_dict.items():
        tf_l[word_l] = count_l / sum_nk
    return tf_l


# подсчет idf-значений для для всех токенов/лемм
def compute_idf(strings_list):
    n = len(strings_list)
    idf_local = dict.fromkeys(strings_list[0].keys(), 0.0)
    for file_tokens in strings_list:
        for word_l, count_l in file_tokens.items():
            if count_l > 0:
                idf_local[word_l] += 1.0

    idf_returned = {}
    for word_l, v in idf_local.items():
        idf_returned[word_l] = math.log(n / float(v))

    return idf_returned


# подсчет tf-idf-значений слов/токенов
def compute_tf_idf_for_file(tf, idf_l):
    tf_idf = dict.fromkeys(tf.keys(), 0)
    for word_l, v in tf.items():
        tf_idf[word_l] = v * idf_l[word_l]
    return tf_idf


if __name__ == '__main__':
    files_number = 100
    directory = 'pages'
    tokens_tf_idf_directory = 'tokens_tf_idf'
    lemmas_tf_idf_directory = 'lemmas_tf_idf'
    save_tokens_path = r'D:\infosearch\tf-idf\tokens-tf-idf'
    save_lemmas_path = r'D:\infosearch\tf-idf\lemmas-tf-idf'
    tokens_file_name = 'tokens.txt'
    lemmas_file_name = 'lemmas.txt'
    # все леммы из всех файлов
    lemmas = []
    # мешок слов для токенов
    bag_of_words_dict = {}
    # мешок слов для лемм
    bag_of_lemmas_dict = {}
    # токены, сгруппированные по номеру документов
    tokens_for_files_dict = {}
    analyzer = pymorphy2.MorphAnalyzer()

    # достаем токены из файла (уже уникальные)
    with open(tokens_file_name, encoding='cp1251', mode='r') as f_tokens:
        tokens = set(f_tokens.read().split('\n')[:-1])

    # достаем леммы из файла (уже уникальные)
    with open('lemmas.txt', encoding='cp1251', mode='r') as f_lemmas:
        for line in f_lemmas:
            lemmas.append(line.split(' ')[0][:-1])

    # формируем мешки слов для токенов и лемм
    # проходим по всем файлам директории 'pages'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # проверка на тип
        if os.path.isfile(file_path):
            # токены из файла (уже уникальные)
            tokens_for_file = tokenize_file(file_path)
            # вытаскиваем номер документа
            file_number = get_file_number(filename)
            tokens_for_files_dict[file_number] = tokens_for_file

            # два пустых вектора мешка слов (ключи есть, значения 0 по умолчанию)
            bag_of_words_vector = dict.fromkeys(tokens, 0)
            bag_of_lemmas_vector = dict.fromkeys(lemmas, 0)
            for word in tokens_for_file:
                parsed_word = analyzer.parse(word)[0]
                lemma = parsed_word.normal_form
                # увеличиваем счетчик соответствующей слову леммы
                if lemma in lemmas:
                    bag_of_lemmas_vector[lemma] += 1
                bag_of_words_vector[word] += 1
            # добавляем векторы к мешкам слов
            bag_of_words_dict[file_number] = bag_of_words_vector
            bag_of_lemmas_dict[file_number] = bag_of_lemmas_vector

    # группируем tf-значения слов по файлам
    tf_tokens_for_files_dict = {}
    tf_lemmas_for_files_dict = {}

    for i in range(1, files_number + 1):
        # tf-значения для конкретного файла
        tf_tokens_i = compute_tf_for_file(bag_of_words_dict[i], tokens_for_files_dict[i])
        tf_lemmas_i = compute_tf_for_file(bag_of_lemmas_dict[i], tokens_for_files_dict[i])

        tf_tokens_for_files_dict[i] = tf_tokens_i
        tf_lemmas_for_files_dict[i] = tf_lemmas_i

    # idf-значения токенов и лемм
    idf_tokens_dict = compute_idf(list(bag_of_words_dict.values()))
    idf_lemmas_dict = compute_idf(list(bag_of_lemmas_dict.values()))

    # группируем tf-idf-значения слов по файлам
    tf_idf_tokens_for_files_dict = {}
    tf_idf_lemmas_for_files_dict = {}

    for i in range(1, files_number + 1):
        tf_idf_tokens_for_files_dict[i] = compute_tf_idf_for_file(tf_tokens_for_files_dict[i], idf_tokens_dict)
        tf_idf_lemmas_for_files_dict[i] = compute_tf_idf_for_file(tf_lemmas_for_files_dict[i], idf_lemmas_dict)

    # запись полученных данных в файлы
    for i in range(1, files_number + 1):
        tokens_newfile_path = os.path.join(save_tokens_path, str(i) + ".txt")
        lemmas_newfile_path = os.path.join(save_lemmas_path, str(i) + ".txt")

        with open(tokens_newfile_path, encoding='cp1251', mode='w+') as f:
            f.write('')

        with open(lemmas_newfile_path, encoding='cp1251', mode='w+') as f:
            f.write('')

        with open(tokens_newfile_path, encoding='cp1251', mode='a') as f:
            for token, count in tf_idf_tokens_for_files_dict[i].items():
                f.write(f'{token} {idf_tokens_dict[token]} {count}\n')

        with open(lemmas_newfile_path, encoding='cp1251', mode='a') as f:
            for lemma, count in tf_idf_lemmas_for_files_dict[i].items():
                f.write(f'{lemma} {idf_lemmas_dict[lemma]} {count}\n')
